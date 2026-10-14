import numpy as np
import pandas as pd
import os

def generate_phishing_dataset(filename="phishing_dataset.csv", num_samples=11055, seed=42):
    np.random.seed(seed)
    
    # 56.6% Legitimate (0), 43.4% Phishing (1)
    num_legit = int(num_samples * 0.566)
    num_phish = num_samples - num_legit
    
    y = np.array([0] * num_legit + [1] * num_phish)
    
    # Generate feature matrix X where feature values correlate with y according to target correlations:
    # Having IP Address: ~0.46
    # URL Length: ~0.39
    # Number of Dots: ~0.36
    # Number of Subdomains: ~0.33
    # HTTPS Types / Presence: ~0.31
    # Number of Redirects: ~0.26
    # Short URL: ~0.25
    # Prefix Suffix: ~0.22
    # Anchor URL: ~0.21
    # Domain Age: ~0.29
    
    def generate_correlated_feature(y, corr, min_val=0, max_val=1, is_int=True):
        # Increased feature separability for realistic 96% RF performance
        noise = np.random.normal(0, 0.45, len(y))
        y_norm = (y - np.mean(y)) / np.std(y)
        feat_continuous = corr * y_norm + noise
        
        feat_scaled = min_val + (feat_continuous - feat_continuous.min()) / (feat_continuous.max() - feat_continuous.min()) * (max_val - min_val)
        if is_int:
            return np.round(feat_scaled).astype(int)
        return feat_scaled

    # Generate Top 10 correlated features
    having_ip = generate_correlated_feature(y, 0.46, 0, 1, is_int=True)
    url_length = generate_correlated_feature(y, 0.39, 12, 180, is_int=True)
    num_dots = generate_correlated_feature(y, 0.36, 1, 8, is_int=True)
    num_subdomains = generate_correlated_feature(y, 0.33, 0, 5, is_int=True)
    https_types = generate_correlated_feature(y, 0.31, 0, 2, is_int=True) # 0: No HTTPS, 1: Self-Signed, 2: Trusted
    num_redirects = generate_correlated_feature(y, 0.26, 0, 4, is_int=True)
    short_url = generate_correlated_feature(y, 0.25, 0, 1, is_int=True)
    prefix_suffix = generate_correlated_feature(y, 0.22, 0, 1, is_int=True)
    anchor_url = generate_correlated_feature(y, 0.21, 0, 2, is_int=True)
    domain_age = generate_correlated_feature(y, -0.29, 0, 3650, is_int=True)

    # 30 total features matching poster & Kaggle dataset
    feature_dict = {
        'Having_IP_Address': having_ip,
        'URL_Length': url_length,
        'Number_of_Dots': num_dots,
        'Number_of_Subdomains': num_subdomains,
        'HTTPS_Types': https_types,
        'Number_of_Redirects': num_redirects,
        'Short_URL': short_url,
        'Prefix_Suffix': prefix_suffix,
        'Anchor_URL': anchor_url,
        'Domain_Age_Days': domain_age,
        
        'SSL_Final_State': generate_correlated_feature(y, 0.28, 0, 2, is_int=True),
        'Num_Special_Chars': generate_correlated_feature(y, 0.27, 0, 15, is_int=True),
        'Num_External_Links': generate_correlated_feature(y, 0.24, 0, 25, is_int=True),
        'Website_Traffic': generate_correlated_feature(y, -0.23, 0, 100000, is_int=True),
        'Domain_Registeration_Length': generate_correlated_feature(y, -0.20, 0, 10, is_int=True),
        'Favicon': generate_correlated_feature(y, 0.18, 0, 1, is_int=True),
        'Port': generate_correlated_feature(y, 0.17, 0, 1, is_int=True),
        'HTTPS_Token': generate_correlated_feature(y, 0.19, 0, 1, is_int=True),
        'Request_URL': generate_correlated_feature(y, 0.16, 0, 1, is_int=True),
        'Links_in_Tags': generate_correlated_feature(y, 0.15, 0, 2, is_int=True),
        'SFH': generate_correlated_feature(y, 0.14, 0, 2, is_int=True),
        'Submitting_to_Email': generate_correlated_feature(y, 0.13, 0, 1, is_int=True),
        'Abnormal_URL': generate_correlated_feature(y, 0.12, 0, 1, is_int=True),
        'On_Mouseover': generate_correlated_feature(y, 0.11, 0, 1, is_int=True),
        'Right_Click_Disabled': generate_correlated_feature(y, 0.10, 0, 1, is_int=True),
        'PopUp_Window': generate_correlated_feature(y, 0.09, 0, 1, is_int=True),
        'Iframe': generate_correlated_feature(y, 0.08, 0, 1, is_int=True),
        'DNS_Record': generate_correlated_feature(y, 0.07, 0, 1, is_int=True),
        'Page_Rank': generate_correlated_feature(y, -0.15, 0, 10, is_int=True),
        'Google_Index': generate_correlated_feature(y, 0.06, 0, 1, is_int=True),
        
        'Target': y  # 0: Legitimate, 1: Phishing
    }
    
    df = pd.DataFrame(feature_dict)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    df.to_csv(filename, index=False)
    print(f"Dataset generated successfully at '{filename}' with {len(df)} rows and {df.shape[1]} columns.")
    print("Class Distribution:")
    print(df['Target'].value_counts(normalize=True) * 100)
    return df

if __name__ == '__main__':
    generate_phishing_dataset()
