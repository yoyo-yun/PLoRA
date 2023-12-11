import os
import time
import torch
import datetime
import evaluate
import numpy as np

from models import *
from tqdm import tqdm
from trainer.optimizer import get_AdamW_cosine_with_hard_restarts, get_AdamW_optim
from trainer.utils import load_dataset_whole_document_plms, Metrics, print_trainable_parameters, get_predictions, load_dataset_whole_document_glove
from trainer.bucket_iteractor import BucketIteratorForFewShot
from cfgs.constants import MODEL_MAP, DATASET_FEW_SHOT_MAP
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ALL_MODLES = {
    "bert": BertForSequenceClassification,
    "roberta": RobertaForSequenceClassification,
    'flan_t5': T5ForConditionalGeneration,
}


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.early_stop = config.TRAIN.early_stop
        self.best_dev_acc = 0
        self.unimproved_iters = 0
        self.iters_not_improved = 0
        self.log_path = self.config.log_path + '/log_run_' + self.config.dataset + '_' + self.config.version + '.txt'
        self.train_metrics = Metrics([evaluate.load(path=f"metrics/{name}.py") for name in config.metrics_list])
        self.dev_metrics = Metrics([evaluate.load(path=f"metrics/{name}.py") for name in config.metrics_list])

    def train(self):
        pass

    def train_epoch(self):
        pass

    def eval(self, eval_itr, state=None):
        pass

    def empty_log(self):
        if (os.path.exists(self.log_path)): os.remove(self.log_path)
        print('Initializing log file ........')
        print('Finished!')
        print('')

    def resume_log(self):
        # Save log information
        logfile = open(self.log_path, 'a+')
        logfile.write(
            'nowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n' +
            'seed:' + str(self.config.seed) +
            '\n'
        )
        logfile.write(str(self.config))
        logfile.write('\n')
        logfile.close()

    def logging(self, log_file, logs):
        logfile = open(
            log_file, 'a+'
        )
        logfile.write(logs)
        logfile.close()

    def get_logging(self, results: dict, eval='training'):
        logs = \
            '==={} phrase...'.format(eval) + "".center(60, " ") + "\n"
        for k, v in results.items():
            logs = logs + k + ": " + "{:.4f} ".format(v)
        logs = logs + "\n"
        return logs

    def ensureDirs(self, *dir_paths):
        for dir_path in dir_paths:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

from trainer.bucket_iteractor import MODEL_MAP, pad_map, _truncate_and_pad, start_map, start_map, end_map
from trainer.utils import DATASET_PATH_MAP


class PLMTrainer(Trainer):
    def __init__(self, config):
        super(PLMTrainer, self).__init__(config)
        # loading dataloader
        self.train_itr, self.dev_itr, self.test_itr, self.usr_stoi, _ = \
            load_dataset_whole_document_plms(self.config, from_sratch=False)
        self.moniter_per_step = len(self.train_itr) // self.config.num_monitor_times_per_epoch
        training_steps_per_epoch = len(self.train_itr) // (self.config.gradient_accumulation_steps)
        self.config.num_train_optimization_steps = self.config.TRAIN.max_epoch * training_steps_per_epoch


    def set_fullshot(self):
        # loading model and plora
        model_name_or_path = MODEL_MAP[self.config.model]
        model = ALL_MODLES[self.config.model].from_pretrained(model_name_or_path,
                                                              return_dict=True,
                                                              num_labels=self.config.num_labels,
                                                              output_hidden_states=True if self.config.mim > 0. else False
                                                              )
        peft_config = PLoraConfig(task_type="SEQ_CLS" if self.config.model not in ["flan_t5"] else "SEQ_2_SEQ_LM",
                                  inference_mode=False,
                                  r=self.config.r,
                                  lora_alpha=self.config.lora_alpha,
                                  lora_dropout=self.config.lora_dropout,
                                  num_virtual_users=len(self.usr_stoi),
                                  user_token_dim=self.config.usr_dim)
        self.net = get_peft_model(model, peft_config).to(self.config.device)

        #set PKI
        # trainable_param_list = ['lora_embedding', 'lora_P', 'lora_B', 'modules_to_save']
        # for n, p in self.net.named_parameters():
        #     if any(key in n for key in trainable_param_list):
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        print_trainable_parameters(self.net, verbose=False)

        # self.optim, self.scheduler = get_AdamW_cosine_with_hard_restarts(self.config, self.net)
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if self.config.model in ['flan_t5'] else None

    def set_stage2(self):
        trainable_param_list = ['lora_embedding', 'lora_P']
        for n, p in self.net.named_parameters():
            if any(key in n for key in trainable_param_list):
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.config.p_rate = 0.0
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        print_trainable_parameters(self.net, verbose=False)
        self.config.mim = 0.

    def train(self, zero_shot=False, start_monitor_epoch=2):
        for epoch in range(1, self.config.TRAIN.max_epoch + 1):
            self.net.train()
            train_results, dev_results = self.train_epoch(epoch, zero_shot, start_monitor_epoch)

            logs = ("    Epoch:{:>2}    ".format(epoch)).center(88, "-") + "".center(70, " ") + '\n' + \
                   self.get_logging(train_results, eval="training")
            print("\r" + logs)

            # logging training logs
            self.logging(self.log_path, logs)

            # logging evaluating logs
            if dev_results is not None:
                eval_logs = self.get_logging(dev_results, eval="evaluating")
                print("\r" + eval_logs)
                self.logging(self.log_path, eval_logs)

                # early stopping
                if dev_results['accuracy'] < self.best_dev_acc:
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.config.TRAIN.patience and self.early_stop == True:
                        early_stop_logs = self.log_path + "\n" + \
                                          "Early Stopping. Epoch: {}, Best Dev Acc: {}".format(epoch, self.best_dev_acc)
                        print(early_stop_logs)
                        self.logging(self.log_path, early_stop_logs)
                        break
                else:
                    self.unimproved_iters = 0
                    self.best_dev_acc = dev_results['accuracy']

    def train_epoch(self, epoch=1, zero_shot=False, start_monitor_epoch=2):
        eval_best_acc = 0.
        eval_best_metrics = None
        epoch_tqdm = tqdm(self.train_itr)
        epoch_tqdm.set_description_str("Processing Epoch: {}".format(epoch))
        self.optim.zero_grad()
        for step, batch in enumerate(epoch_tqdm):
            self.net.train()
            if zero_shot: batch['p'] = None
            p_mask = self.get_personality_mask(self.config.p_rate, batch['p'])
            batch['p_mask'] = p_mask
            batch['mim'] = self.config.mim
            outputs = self.net(**batch)
            logits = outputs.logits
            loss = outputs.loss
            if batch['mim'] > 0 and not zero_shot:
                # detach generic model
                with torch.no_grad():
                    batch['mim'] = -1
                    batch['p_mask'] = self.get_personality_mask(1., batch['p'])
                    outputs_ = self.net(**batch)
                    pooled_outputs_ = outputs_.pooled_output
                logits_ = self.net.classifier(pooled_outputs_.detach())

                # if p_mask is None:
                #     logits_ = torch.einsum("bd,b->bd", logits_, p_mask)
                #     logits = torch.einsum("bd,b->bd", logits, p_mask)

                # # # traditional KD method
                # outputs_ = self.net(**batch)
                # logits_ = outputs_.logits

                # mim_loss = torch.nn.MSELoss()(logits_, logits)
                mim_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)\
                    (torch.log_softmax(logits_ / 4, dim=-1), torch.softmax(logits / 4, dim=-1))
                loss += mim_loss * self.config.mim
                # loss += mim_loss * 1
            if self.config.gradient_accumulation_steps >= 2:
                loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optim.step()
                if self.scheduler is not None: self.scheduler.step()
                self.optim.zero_grad()

            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["labels"])
            self.train_metrics.add_batch(
                predictions=predictions,
                references=references,
            )

            if step % self.moniter_per_step == 0 and step != 0 and epoch >= start_monitor_epoch:
                self.net.eval()
                with torch.no_grad():
                    dev_metrics = self.eval(self.dev_itr, zero_shot)

                # monitoring eval metrics
                if dev_metrics["accuracy"] > eval_best_acc:
                    eval_best_acc = dev_metrics["accuracy"]
                    eval_best_metrics = dev_metrics
                    if dev_metrics["accuracy"] > self.best_dev_acc:
                        # saving models
                        self.best_dev_acc = dev_metrics["accuracy"]
                        self.saving_model()

        return self.train_metrics.compute(average="macro"), eval_best_metrics

    def get_personality_mask(self, p_rate, p):
        return torch.from_numpy(np.random.binomial(1, 1 - p_rate, p.shape[0])).to(p.device) if p is not None else None

    def saving_model(self):
        SAVED_MODEL_PATH = self.config.ckpts_path
        self.ensureDirs(os.path.join(SAVED_MODEL_PATH, self.config.dataset, self.config.model))
        self.net.save_pretrained(os.path.join(SAVED_MODEL_PATH, self.config.dataset, self.config.model))

    def load_state(self, dataset):
        SAVED_MODEL_PATH = self.config.ckpts_path
        path = os.path.join(SAVED_MODEL_PATH, dataset, self.config.model)
        config = PLoraConfig.from_pretrained(path)
        model = ALL_MODLES[self.config.model].from_pretrained(config.base_model_name_or_path,
                                                              return_dict=True, num_labels=self.config.num_labels)
        net = PeftModel.from_pretrained(model, path)
        self.net = net.to(self.config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[self.config.model]) \
            if self.config.model in ['flan_t5'] else None

    def eval(self, eval_itr, zero_shot=False):
        self.net.eval()
        for step, batch in enumerate(eval_itr):
            if zero_shot: batch['p'] = None
            outputs = self.net(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = get_predictions(self.tokenizer, predictions), get_predictions(self.tokenizer,
                                                                                                    batch["labels"])
            self.dev_metrics.add_batch(
                predictions=predictions,
                references=references,
            )
        return self.dev_metrics.compute(average="macro")

    def set_fewshot(self, shot=-1):
        self.load_state(DATASET_FEW_SHOT_MAP[self.config.dataset])
        # replace original full-shot learning users with few-shot learning users
        self.net.base_model.lora_embedding = torch.nn.Embedding(
            len(self.usr_stoi), self.config.usr_dim,
            _weight=torch.zeros(len(self.usr_stoi), self.config.usr_dim)).to(self.net.device)
        trainable_param_list = ['lora_embedding']
        for n, p in self.net.named_parameters():
            if any(key in n for key in trainable_param_list):
                p.requires_grad = True
            else:
                p.requires_grad = False
        # self.optim, self.scheduler = get_AdamW_cosine_with_hard_restarts(self.config, self.net)
        # self.config.TRAIN.lr_base = self.config.TRAIN.lr_base * 10.
        self.config.mim = 0.
        self.config.p_rate = 0.0
        self.optim, self.scheduler = get_AdamW_optim(self.config, self.net)
        print_trainable_parameters(self.net)

        if shot > 0:
            self.train_itr = BucketIteratorForFewShot(data=self.train_itr,
                                                      batch_size=self.config.TRAIN.batch_size,
                                                      plm_name=self.config.model,
                                                      device=self.config.device,
                                                      shot_number=shot)
            self.moniter_per_step = len(self.train_itr) - 1

    def set_zeroshot(self):
        self.load_state(DATASET_FEW_SHOT_MAP[self.config.dataset])

    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log()
            self.resume_log()
            self.set_fullshot()
            self.train(zero_shot=True) if self.config.nop else self.train(zero_shot=False)
            metircs = self.eval(self.test_itr, zero_shot=True) if self.config.nop else self.eval(self.test_itr, zero_shot=False)
            print(metircs)
        if run_mode == 'fewshot':
            self.empty_log()
            self.resume_log()
            self.set_fewshot(self.config.few)
            self.train(start_monitor_epoch=1)
            metircs = self.eval(self.test_itr, zero_shot=False)
            print(metircs)
        if run_mode == 'zeroshot':
            self.set_zeroshot()
            metircs = self.eval(self.dev_itr, zero_shot=True)
            metircs = self.eval(self.test_itr, zero_shot=True)
            print(metircs)
        if run_mode == 'twostage':
            # self.empty_log()
            # self.resume_log()
            # self.set_fullshot()
            # print("=== This is the first stage....")
            # self.train(zero_shot=True)
            self.load_state(self.config.dataset)
            self.set_stage2()
            print("=== This is the second stage....")
            self.train(zero_shot=False, start_monitor_epoch=1)
        elif run_mode == 'val':
            self.load_state(self.config.dataset)
            metircs = self.eval(self.dev_itr, zero_shot=False)
            print(metircs)
        elif run_mode == 'test':
            self.load_state(self.config.dataset)
            metircs = self.eval(self.test_itr, zero_shot=False)
            print(metircs)
        else:
            exit(-1)