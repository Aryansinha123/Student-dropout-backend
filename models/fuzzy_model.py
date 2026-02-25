def fuzzy_risk(data_dict):
    score = 0

    # Academic performance
    if data_dict["Curricular units 1st sem (approved)"] < 3:
        score += 20

    if data_dict["Curricular units 2nd sem (approved)"] < 3:
        score += 20

    if data_dict["Curricular units 1st sem (grade)"] < 10:
        score += 15

    if data_dict["Curricular units 2nd sem (grade)"] < 10:
        score += 15

    # Admission strength
    if data_dict["Admission grade"] < 120:
        score += 10

    # Financial factors
    if data_dict["Debtor"] == 1:
        score += 10

    if data_dict["Tuition fees up to date"] == 0:
        score += 10

    return min(score, 100)