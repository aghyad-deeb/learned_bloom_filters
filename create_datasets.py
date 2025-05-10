import os
import random
import string
import pandas as pd

## paper specefications:
# negative set in the paper is a mix of random URLs and whitelisted
# URLs that can be mistaken for phishing pages.

## Actual data:
# the training dataset is adopted from https://ieee-dataport.org/documents/phishing-attack-dataset#files
# negative dataset filtered for benign from https://data.mendeley.com/datasets/vfszbj9b36/1
# in addition to randomly generated urls


def get_negative_dataset_real(raw_data_dir, negative_dataset_name, benign_column_negative_dataset, benign_keyword_negative_dataset):
    dataset_path = os.path.join(raw_data_dir, negative_dataset_name)
    df = pd.read_csv(dataset_path)
    return df[df[benign_column_negative_dataset] == benign_keyword_negative_dataset]


def generate_realistic_random_url(
    protocols=None, 
    tlds=None, 
    file_extensions=None, 
    path_patterns=None, 
    subdomain_prob=0.3, 
    path_prob=0.6, 
    common_path_prob=0.5, 
    query_prob=0.2,
    subdomain_length=(2, 8),
    domain_length=(6, 12),
    path_length=(4, 12),
    param_value_length=(16, 32),
    max_params=3
):
    """
    Generate a realistic random URL with customizable components.

    Args:
        protocols: List of protocol strings to choose from (e.g., http://).
        tlds: List of top-level domains (e.g., .com).
        file_extensions: List of file extensions for paths (e.g., .html).
        path_patterns: List of common path patterns (e.g., images/secure).
        subdomain_prob: Probability of including a subdomain (e.g., the mail part in mail.google.com).
        path_prob: Probability of adding a path to the URL.
        common_path_prob: Probability of using a path from `path_patterns`.
        query_prob: Probability of adding query parameters.
        subdomain_length: (min, max) length for subdomain.
        domain_length: (min, max) length for main domain.
        path_length: (min, max) length for random path.
        param_value_length: (min, max) length for parameter values.
        max_params: Maximum number of query parameters.

    Returns:
        str: A generated URL string.
    """
    # Lists of components to create realistic-looking URLs
    if protocols is None:
        protocols = ['http://', 'https://']
    if tlds is None:
        tlds = ['.com', '.org', '.net', '.ru', '.us', '.co', '.site', '.app']
    if file_extensions is None:
        file_extensions = ['.php', '.html', '.aspx', '']
    if path_patterns is None:
        path_patterns = [
            'checkpoint/login',
            'images/secure',
            'admin/panel',
            'verify/account',
            'secure/login',
            'manager/auth'
        ]
    
    # Generate random components
    protocol = random.choice(protocols)
    
    # Create domain (might include subdomain)
    if random.random() < subdomain_prob:
        subdomain = ''.join(random.choices(string.ascii_lowercase, k=random.randint(*subdomain_length)))
        domain = f"{subdomain}."
    else:
        domain = ""
    
    # Main domain name
    domain += ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(*domain_length)))
    
    # TLD
    tld = random.choice(tlds)
    
    # Build the base URL
    url = f"{protocol}{domain}{tld}"
    
    # Add path and parameters
    if random.random() < path_prob:
        # Random path
        if random.random() < common_path_prob:
            path = random.choice(path_patterns)
        else:
            path = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(*path_length)))
        
        # Add file extension
        path += random.choice(file_extensions)
        
        url += f"/{path}"
        
        # Add query parameters
        if random.random() < query_prob:
            params = []
            num_params = random.randint(1, max_params)
            param_names = ['id', 'session', 'token', 'auth', 'cmd', 'user']
            for _ in range(num_params):
                param_name = random.choice(param_names)
                param_value = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(*param_value_length)))
                params.append(f"{param_name}={param_value}")
            url += '?' + '&'.join(params)
    
    return url

def main() :
    random_state = 42
    raw_data_dir = "raw_data"
    training_dataset_name = "phishing_attack_big_train.csv"
    negative_dataset_name = "mandeley_phishing_and_benign.csv"

    benign_column_negative_dataset = "type"
    benign_keyword_negative_dataset = "legitimate"
    negative_dataset_real = get_negative_dataset_real(
        raw_data_dir, negative_dataset_name,
        benign_column_negative_dataset, benign_keyword_negative_dataset
    )
    negative_dataset_real["real_or_random"] = "real"
    negative_dataset_real["type"] = "benign"
    
    num_rows_negative_dataset_real = negative_dataset_real.shape[0]
    # make the same number of random_generated_urls as the real ones
    randomly_generated_urls = [generate_realistic_random_url() for _ in range(num_rows_negative_dataset_real)]
    negative_dataset_dummies = pd.DataFrame({
        "url": randomly_generated_urls,
        "type": "benign",
        "real_or_random": "random"
    })
    negative_dataset = pd.concat(
        [negative_dataset_real, negative_dataset_dummies], ignore_index=True
    )

    # Shuffle the dataset before saving
    negative_dataset = (
        negative_dataset
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
    )

    processed_data_dir = "data"
    negative_dataset.to_csv(
        os.path.join(processed_data_dir, "negative_dataset.csv"), index=False
    )

    # Read the training dataset (phishing_attack_big_train.csv)
    # This file is tab-delimited with no header row and two columns: type and url
    # Some lines may have inconsistent formatting
    training_dataset_path = os.path.join(raw_data_dir, training_dataset_name)
    
    # Read with tab delimiter, add column names, and handle parsing errors
    training_dataset = pd.read_csv(
        training_dataset_path,
        sep='\t',
        header=None,
        names=["type", "url"],
        on_bad_lines='skip',  # Skip lines with incorrect format
        engine='python'  # More flexible parsing engine
    )
    
    # Remove any rows with NaN values after parsing
    training_dataset = training_dataset.dropna()
    
    # Make sure we only have two valid types
    training_dataset = training_dataset[training_dataset["type"].isin(["legitimate", "phishing"])]
    
    # Map "legitimate" to "benign" to maintain consistent naming
    training_dataset["type"] = training_dataset["type"].map(
        {"legitimate": "benign", "phishing": "phishing"}
    )
    
    # Check class distribution before balancing
    class_counts_before = training_dataset["type"].value_counts()
    print(f"Class distribution before balancing: {class_counts_before}")
    
    # Balance the dataset by undersampling the majority class
    benign_samples = training_dataset[training_dataset["type"] == "benign"]
    phishing_samples = training_dataset[training_dataset["type"] == "phishing"]
    
    # Get the count of the minority class
    min_class_count = min(len(benign_samples), len(phishing_samples))
    
    # Undersample the majority class to match the minority class
    if len(benign_samples) > min_class_count:
        benign_samples = benign_samples.sample(min_class_count, random_state=random_state)
    if len(phishing_samples) > min_class_count:
        phishing_samples = phishing_samples.sample(min_class_count, random_state=random_state)
    
    # Combine the balanced classes
    balanced_training_dataset = pd.concat([benign_samples, phishing_samples])
    
    # Verify the balance
    class_counts_after = balanced_training_dataset["type"].value_counts()
    print(f"Class distribution after balancing: {class_counts_after}")
    
    # Shuffle the balanced dataset
    balanced_training_dataset = (
        balanced_training_dataset
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
    )
    
    # Save the processed balanced training dataset
    balanced_training_dataset.to_csv(
        os.path.join(processed_data_dir, "training_dataset_extended_balanced.csv"), index=False
    )
    
    # Print type counts for verification
    print(f"{balanced_training_dataset['type'].value_counts()=}")
    

if __name__ == "__main__":
    main()
