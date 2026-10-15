---
license: mit
tags: [steganalysis, text-steganography, lightgbm, albert, shap, explainable-ai]
---
# Text Steganography Detector + Reveal

4-class detector (**Clean / ZeroWidth / Whitespace / Synonym**) with SHAP explanations and automatic payload extraction.
Trained on IMDB cover text; externally validated on WikiText-2 and SMS Spam.

## Results (IMDB test, grouped split - no cover-text leakage)
```
                Accuracy  Precision  Recall      F1     AUC  Params (nodes/weights)   MACs/sample  Latency ms/sample (end-to-end)
LightGBM-Lite     0.9677     0.9677  0.9677  0.9677  0.9977                  9794.0  1.452281e+03                          0.3923
LightGBM-TFIDF    0.9822     0.9823  0.9822  0.9822  0.9993                 20438.0  2.564774e+03                          1.6366
ALBERT            0.9959     0.9959  0.9959  0.9959  0.9998              11686660.0  1.407773e+10                         11.3753
```
## External validation
```
         Model     Dataset  Accuracy  MacroF1
 LightGBM-Lite IMDB (test)    0.9677   0.9677
LightGBM-TFIDF IMDB (test)    0.9822   0.9822
        ALBERT IMDB (test)    0.9959   0.9959
 LightGBM-Lite  WikiText-2    0.9601   0.9601
LightGBM-TFIDF  WikiText-2    0.9606   0.9606
        ALBERT  WikiText-2    0.9160   0.9147
 LightGBM-Lite    SMS Spam    0.9758   0.9758
LightGBM-TFIDF    SMS Spam    0.9879   0.9879
        ALBERT    SMS Spam    0.8458   0.8296
```
## Use
```python
from detector import StegoDetector
det = StegoDetector('.', use_albert=False)      # LightGBM-Lite + exact TreeSHAP, CPU, ~ms
print(det.analyze(text))                         # label, probs, revealed payload, top SHAP features
```
Local app: `pip install -r requirements.txt && python app.py`

**Limitations:** the reveal step only decodes the three implemented channels (known encoding rules); detection is blind/statistical.
