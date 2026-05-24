def calculate_confidence_score(predicted_time):
    """Calculate confidence score from predicted delivery time."""
    return max(65, min(98, 100 - (predicted_time / 2)))


def calculate_risk_level(predicted_time):
    """Derive risk level label from predicted delivery time."""
    if predicted_time < 25:
        return "🟢 Low Delay Risk"
    elif predicted_time < 45:
        return "🟡 Medium Delay Risk"
    else:
        return "🔴 High Delay Risk"
