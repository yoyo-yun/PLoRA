# This code is clone from peft third package and being changed for personalization
__version__ = "0.3.0.dev0"
from models.ppeft.mapping import get_peft_model
from models.ppeft.plora import PLoraConfig, PeftType, PLoraModel
# from models.ppeft.pprompt_tuning import PromptTuningConfig
from models.ppeft.peft_model import PeftModel
