def hybrid_score(fuzzy_score, ann_probability):

    ann_score = ann_probability * 100

    # If ANN very confident (<20 or >80), trust ANN
    if ann_score < 20:
        final_score = ann_score * 0.8 + fuzzy_score * 0.2
    elif ann_score > 80:
        final_score = ann_score * 0.8 + fuzzy_score * 0.2
    else:
        # If ANN uncertain, use fuzzy more
        final_score = ann_score * 0.6 + fuzzy_score * 0.4

    if final_score > 75:
        category = "High"
    elif final_score > 50:
        category = "Medium"
    else:
        category = "Low"

    return final_score, category