import pandas as pd
import numpy as np
import random

def generate_11k_urls_dataset(output_file="kaggle_phishing_dataset_with_urls.csv", num_records=11055, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    phishing_domains = [
        "fake-login-security", "paypal-account-update", "appleid-verify-user", "secure-bank-login",
        "google-drive-share-doc", "netflix-billing-update", "amazon-order-refund-check", "chase-online-verify",
        "wellsfargo-security-alert", "facebook-checkpoint-verify", "instagram-badge-verify", "microsoft-office365-login",
        "binance-withdraw-confirm", "coinbase-auth-update", "meta-business-support", "dhl-tracking-parcel",
        "usps-redelivery-confirm", "fedex-shipment-alert", "dropbox-file-download", "linkedin-career-offer"
    ]
    
    phishing_tlds = [".com", ".net", ".org", ".xyz", ".info", ".online", ".site", ".top", ".cc", ".tk", ".club"]
    
    legit_brands = [
        "google", "apple", "microsoft", "amazon", "github", "wikipedia", "youtube", "linkedin",
        "facebook", "instagram", "twitter", "reddit", "netflix", "adobe", "dropbox", "salesforce",
        "quora", "spotify", "zoom", "stackoverflow", "medium", "paypal", "ebay", "pinterest"
    ]
    
    legit_tlds = [".com", ".org", ".net", ".edu", ".gov", ".io"]
    
    urls = []
    targets = []
    
    num_legit = int(num_records * 0.566)
    num_phish = num_records - num_legit
    
    # Generate 56.6% Legitimate URLs (Target 0)
    for i in range(num_legit):
        brand = random.choice(legit_brands)
        tld = random.choice(legit_tlds)
        sub = random.choice(["", "www.", "support.", "docs.", "blog.", "mail."])
        path = random.choice(["", "/home", "/about", "/search?q=info", "/products/item", "/login", "/help/index.html"])
        url = f"https://{sub}{brand}{tld}{path}"
        urls.append(url)
        targets.append(0)
        
    # Generate 43.4% Phishing URLs (Target 1)
    for i in range(num_phish):
        phish_style = random.randint(1, 4)
        if phish_style == 1:
            # IP Address
            ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            url = f"http://{ip}/login/verify.php?user_id={random.randint(1000, 9999)}"
        elif phish_style == 2:
            # Shortener
            shortener = random.choice(["bit.ly", "tinyurl.com", "is.gd", "t.co"])
            url = f"http://{shortener}/{random.choice(phishing_domains)}-{random.randint(100,999)}"
        elif phish_style == 3:
            # Hyphen & Subdomain Spoofing
            domain = random.choice(phishing_domains)
            tld = random.choice(phishing_tlds)
            url = f"http://{domain}-{random.randint(10,99)}{tld}/account/login.html"
        else:
            # Long URL with multiple parameters
            brand = random.choice(phishing_domains)
            url = f"http://www.{brand}.info/verify-identity/update-security-account?token={random.randint(100000, 999999)}&ref=session"
            
        urls.append(url)
        targets.append(1)
        
    # Combine & Shuffle
    df_raw = pd.DataFrame({
        'Record_ID': range(1, num_records + 1),
        'Website_URL': urls,
        'Target_Class': targets,
        'Label': ['Legitimate (0)' if t == 0 else 'Phishing (1)' for t in targets]
    }).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Re-assign Record IDs after shuffle
    df_raw['Record_ID'] = range(1, num_records + 1)
    
    df_raw.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(df_raw)} URLs (Legitimate: {num_legit}, Phishing: {num_phish})")
    return df_raw

if __name__ == '__main__':
    generate_11k_urls_dataset()
