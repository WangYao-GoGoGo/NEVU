class TitleNlpNer:
    def __init__(self, en_text, en_type, pos_type):
        self.en_text = en_text
        self.en_type = en_type
        self.pos_type = pos_type

    def to_dict(self):
        return {
            "en_text": self.en_text,
            "en_type": self.en_type,
            "pos_type": self.pos_type
        }

class ContentNlpNer:
    def __init__(self, en_text, en_type, pos_type):
        self.en_text = en_text
        self.en_type = en_type
        self.pos_type = pos_type

    def to_dict(self):
        return {
            "en_text": self.en_text,
            "en_type": self.en_type,
            "pos_type": self.pos_type
        }
class ImgCvDetBBox:
    def __init__(self, fir_po, se_po, th_po, fo_po):
        self.fir_po = fir_po
        self.se_po = se_po
        self.th_po = th_po
        self.fo_po = fo_po

    def to_dict(self):
        return [self.fir_po, self.se_po, self.th_po, self.fo_po]

    # def to_dict(self):
    #     return {
    #         "lu": self.fir_po,
    #         "ru": self.se_po,
    #         "lb": self.th_po,
    #         "rb": self.fo_po
    #     }


class Doc:
    def __init__(self, guid, imgpth, time, title, content, imgdes, title_nlpners, content_nlpners, img_cvdetbboxs, img_caps):
        self.guid = guid
        self.imgpth = imgpth
        self.time = time
        self.title = title
        self.content = content
        self.imgdes = imgdes
        self.title_nlpners = title_nlpners
        self.content_nlpners = content_nlpners
        self.img_cvdetbboxs = img_cvdetbboxs
        self.img_caps = img_caps

    def to_dict(self):
        title_nlpners_json = [title_nlpner.to_dict() for title_nlpner in self.title_nlpners]
        content_nlpners_json = [content_nlpner.to_dict() for content_nlpner in self.content_nlpners]
        img_cvdetbboxs_json = [img_cvdetbbox.to_dict() for img_cvdetbbox in self.img_cvdetbboxs]
        return {
            "guid": self.guid,
            "imgpth": self.imgpth,
            "time": self.time,
            "title": self.title,
            "content": self.content,
            "imgdes": self.imgdes,
            "title_nlpners": title_nlpners_json,
            "content_nlpners": content_nlpners_json,
            "img_cvdetbboxs": img_cvdetbboxs_json,
            "img_caps": self.img_caps
        }
