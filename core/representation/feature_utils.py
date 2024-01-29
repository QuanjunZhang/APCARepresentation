import json
import os

"""
使用工具提取特征
"""


def features(coming_path=None, add_path=None):
    results = {
        'code_features': dict(),
        'context_features': dict(),
        'pattern-features': dict(),
    }
    if coming_path is not None and os.path.exists(coming_path):
        results['code_features'], results['context_features'] = coming_features(coming_path)
    if add_path is not None and os.path.exists(add_path):
        results['pattern-features'] = add_feature(add_path)
    return results


def coming_features(path):
    feature1, feature2 = dict(), dict()
    with open(path, 'r') as f:
        t = json.load(f)
        feature_list1 = t['probability']
        feature_list2 = t['probabilityParent']
        for feature in feature_list1:
            if feature['c'] not in feature1:
                feature1[feature['c']] = feature['f']
        for feature in feature_list2:
            if feature['c'] not in feature2:
                feature2[feature['c']] = feature['f']
        f.close()
    return feature1, feature2


def add_feature(path):
    with open(path, 'r') as f:
        patterns = json.load(f)['repairPatterns']
        patterns = {k: patterns[k] for k in patterns.keys() if patterns[k] > 0}
        if len(patterns) > 0:
            svalue = sum(patterns.values())
            for k in patterns.keys():
                patterns[k] = patterns[k] / svalue
        f.close()
    return patterns

