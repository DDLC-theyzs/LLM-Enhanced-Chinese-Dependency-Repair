# nlp_parser.py

"""
封装hanlp,用于依存句法的初步分析。

依赖:
    pip install hanlp -U

"""

import hanlp

class HanLPMultiTaskParser:
    def __init__(self):
        hanlp.pretrained.mtl.ALL
        self.parser = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

    def parse(self, sentence: str):
        doc = self.parser(sentence, tasks='dep').to_conll() 
        return doc
