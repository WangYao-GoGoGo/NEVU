
class TextEvent:
    def __init__(self, trigger_text, trigger_type, arguments: Argument, argument_role):
        self.trigger_text = trigger_text
        self.trigger_type = trigger_type
        self.argument = argument
        self.argument_role = argument_role

class ImgEvent:
    def __init__(self, trigger_text, trigger_type, argument, argument_role, img_box):
        self.trigger_text = trigger_text
        self.trigger_type = trigger_type
        self.argument = argument
        self.argument_role = argument_role
        self.img_box = img_box

class MultimodalEvent:
    def __init__(self, trigger_text, trigger_type, argument, argument_role, img_box):
        self.trigger_text = trigger_text
        self.trigger_type = trigger_type
        self.argument = argument
        self.argument_role = argument_role
        self.img_box = img_box

class Argument:
    def __init__(self, arg_st, arg_ed, arg_type):
        self.arg_st = arg_st
        self.arg_ed = arg_ed
        self.arg_type = arg_type



class paragraph:
    def __init__(self, imgpth, time, title, content, imgdes, title_ner, content_ner, img_box, image_cap, img_obj,
                 TextEvent, ImgEvent, MultimodalEvent):
        self.imgpth = imgpth
        self.time = time
        self.title = title
        self.content = content
        self.imgdes = imgdes
        self.title_ner = title_ner
        self.content_ner = content_ner
        self.img_box = img_box
        self.image_cap = image_cap
        self.img_obj = img_obj
        self.TextEvent = TextEvent
        self.ImgEvent = ImgEvent
        self.MultimodalEvent = MultimodalEvent


