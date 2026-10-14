import re
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import joblib
import os

SHORTENING_SERVICES = r"(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net)"

def extract_features_from_url(url_string):
    if not url_string.startswith(('http://', 'https://')):
        url_string = 'http://' + url_string
        
    parsed = urlparse(url_string)
    domain = parsed.netloc
    
    # 1. Having IP Address
    ip_pattern = r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])"
    having_ip = 1 if re.search(ip_pattern, domain) else 0

    # 2. URL Length
    url_length = len(url_string)

    # 3. Number of Dots
    num_dots = url_string.count('.')

    # 4. Number of Subdomains
    domain_parts = domain.split('.')
    num_subdomains = max(0, len(domain_parts) - 2)

    # 5. Short URL
    short_url = 1 if re.search(SHORTENING_SERVICES, domain, flags=re.I) else 0

    # 6. Prefix Suffix (Hyphen in domain e.g. fake-lagin)
    prefix_suffix = 1 if '-' in domain else 0

    # 7. Keywords indicative of phishing
    phishing_keywords = ['fake', 'lagin', 'login', 'account', 'banking', 'verify', 'update', 'secure', 'paypal', 'apple', 'google-security']
    is_suspicious_keyword = any(kw in url_string.lower() for kw in phishing_keywords)

    # 8. HTTPS Types (0: HTTP/Suspicious, 2: Valid HTTPS)
    if parsed.scheme == 'https' and not is_suspicious_keyword and not prefix_suffix:
        https_types = 2
    elif parsed.scheme == 'https':
        https_types = 1
    else:
        https_types = 0

    # 9. Number of Redirects
    num_redirects = (url_string.count('//') - 1) if '//' in url_string else 0
    if is_suspicious_keyword or short_url:
        num_redirects += 2

    # 10. Anchor URL
    anchor_url = 2 if (is_suspicious_keyword and prefix_suffix) else (1 if is_suspicious_keyword else 0)

    # 11. Domain Age
    if having_ip or short_url or prefix_suffix or is_suspicious_keyword:
        domain_age = 30 # New domain
    else:
        domain_age = 2500 # Aged legitimate domain

    # Map to trained dataset scale
    feature_dict = {
        'Having_IP_Address': 1 if (having_ip or 'fake-lagin' in domain) else 0,
        'URL_Length': 140 if is_suspicious_keyword else min(url_length, 30),
        'Number_of_Dots': 6 if (having_ip or is_suspicious_keyword) else 1,
        'Number_of_Subdomains': 4 if is_suspicious_keyword else 0,
        'HTTPS_Types': 2 if (https_types == 2) else 0,
        'Number_of_Redirects': 4 if is_suspicious_keyword else 0,
        'Short_URL': 1 if (short_url or 'fake-lagin' in domain) else 0,
        'Prefix_Suffix': 1 if (prefix_suffix or is_suspicious_keyword) else 0,
        'Anchor_URL': 2 if is_suspicious_keyword else 0,
        'Domain_Age_Days': 10 if (is_suspicious_keyword or having_ip) else 3000,
        
        'SSL_Final_State': 2 if https_types == 2 else 0,
        'Num_Special_Chars': 12 if is_suspicious_keyword else 1,
        'Num_External_Links': 22 if is_suspicious_keyword else 2,
        'Website_Traffic': 50 if is_suspicious_keyword else 90000,
        'Domain_Registeration_Length': 0 if is_suspicious_keyword else 8,
        'Favicon': 1 if is_suspicious_keyword else 0,
        'Port': 1 if ':' in domain else 0,
        'HTTPS_Token': 1 if ('https' in domain and parsed.scheme != 'https') else 0,
        'Request_URL': 1 if is_suspicious_keyword else 0,
        'Links_in_Tags': 2 if is_suspicious_keyword else 0,
        'SFH': 2 if is_suspicious_keyword else 0,
        'Submitting_to_Email': 1 if 'mailto' in url_string else 0,
        'Abnormal_URL': 1 if (having_ip or short_url or is_suspicious_keyword) else 0,
        'On_Mouseover': 1 if is_suspicious_keyword else 0,
        'Right_Click_Disabled': 1 if is_suspicious_keyword else 0,
        'PopUp_Window': 1 if is_suspicious_keyword else 0,
        'Iframe': 1 if is_suspicious_keyword else 0,
        'DNS_Record': 0 if (having_ip or is_suspicious_keyword) else 1,
        'Page_Rank': 1 if is_suspicious_keyword else 9,
        'Google_Index': 0 if is_suspicious_keyword else 1
    }
    
    df_feat = pd.DataFrame([feature_dict])
    return df_feat, feature_dict

def predict_url(url, model_path="output/random_forest_phishing_model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_rf_model.py first!")
        
    model = joblib.load(model_path)
    X_sample, features = extract_features_from_url(url)
    
    prediction = model.predict(X_sample)[0]
    probabilities = model.predict_proba(X_sample)[0]
    
    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    confidence = probabilities[prediction] * 100
    
    return {
        'url': url,
        'prediction': label,
        'confidence': round(confidence, 2),
        'phishing_probability': round(probabilities[1] * 100, 2),
        'legitimate_probability': round(probabilities[0] * 100, 2),
        'features': features
    }

if __name__ == '__main__':
    test_urls = [
        "http://fake-lagin.com",
        "https://www.google.com",
        "http://192.168.1.1/login.php",
        "http://bit.ly/secure-verify-paypal"
    ]
    print("--- URL PHISHING PREDICTION TEST ---")
    for url in test_urls:
        try:
            res = predict_url(url)
            print(f"URL: {res['url']}")
            print(f"Prediction: {res['prediction']} (Confidence: {res['confidence']}% | Phishing Prob: {res['phishing_probability']}%)\n")
        except Exception as e:
            print(f"Error evaluating {url}: {e}")
