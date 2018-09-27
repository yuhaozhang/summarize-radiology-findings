"""
Scoring with ROUGE metrics.
"""
from pythonrouge.pythonrouge import Pythonrouge

def get_rouge(hypotheses, reference, sent_split=True, use_cf=True):
    assert len(hypotheses) == len(reference)
    assert len(hypotheses) > 0
    
    hyps = []
    refs = []
    # prepare
    for hyp, ref in zip(hypotheses, reference):
        hyp = " ".join(hyp)
        ref = " ".join(ref)
        if sent_split:
            hs = [x.strip() for x in hyp.split('.') if len(x.strip()) > 0]
            rs = [x.strip() for x in ref.split('.') if len(x.strip()) > 0]
            hyps += [hs]
            refs += [[rs]]
        else:
            hyps += [[hyp]]
            refs += [[[ref]]]
    print("Calculating ROUGE...")
    rouge = Pythonrouge(summary_file_exist=False, summary=hyps, reference=refs, \
            n_gram=2, ROUGE_SU4=False, ROUGE_L=True, recall_only=False, stemming=False, stopwords=False,\
            word_level=True, length_limit=False, use_cf=use_cf, cf=95, scoring_formula='average', \
            resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    print("ROUGE done.")

    r1 = score['ROUGE-1-F']*100
    r2 = score['ROUGE-2-F']*100
    rl = score['ROUGE-L-F']*100
    if not use_cf:
        return r1, r2, rl
    else:
        r1_cf = [x*100 for x in score['ROUGE-1-F-cf95']]
        r2_cf = [x*100 for x in score['ROUGE-2-F-cf95']]
        rl_cf = [x*100 for x in score['ROUGE-L-F-cf95']]
        return r1, r2, rl, r1_cf, r2_cf, rl_cf
