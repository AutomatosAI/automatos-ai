"""
Enhanced Logo Downloader - Multiple Sources
"""
import requests
import time
from pathlib import Path
import json
from urllib.parse import quote

# Read failed logos
with open('failed_logos.json', 'r') as f:
    failed_brands = json.load(f)

print(f"📋 Retrying {len(failed_brands)} failed logos with alternative methods\n")

output_dir = Path("frontend/public/logos")
output_dir.mkdir(parents=True, exist_ok=True)

success = []
still_failed = []

for i, item in enumerate(failed_brands, 1):
    brand = item['brand']
    domain = item['domain']
    
    print(f"[{i}/{len(failed_brands)}] {brand:30s} ", end="")
    
    # Method 1: Try logo.dev (different API)
    try:
        url = f"https://img.logo.dev/{domain}?size=200&format=png"
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and len(response.content) > 1000:
            output_path = output_dir / f"{brand}.png"
            output_path.write_bytes(response.content)
            print("✅ logo.dev")
            success.append(brand)
            time.sleep(0.5)
            continue
    except:
        pass
    
    # Method 2: Try brandfetch
    try:
        url = f"https://api.brandfetch.io/v2/brands/{domain}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'logos' in data and len(data['logos']) > 0:
                logo_url = data['logos'][0].get('formats', [{}])[0].get('src')
                if logo_url:
                    logo_response = requests.get(logo_url, timeout=10)
                    if logo_response.status_code == 200:
                        output_path = output_dir / f"{brand}.png"
                        output_path.write_bytes(logo_response.content)
                        print("✅ brandfetch")
                        success.append(brand)
                        time.sleep(0.5)
                        continue
    except:
        pass
    
    # Method 3: Try simple domain/favicon approach
    try:
        url = f"https://{domain}/favicon.ico"
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and len(response.content) > 500:
            output_path = output_dir / f"{brand}.png"
            output_path.write_bytes(response.content)
            print("✅ favicon")
            success.append(brand)
            time.sleep(0.5)
            continue
    except:
        pass
    
    print("❌ All methods failed")
    still_failed.append(item)
    time.sleep(0.5)

print(f"\n{'='*60}")
print(f"📊 RETRY SUMMARY")
print(f"{'='*60}")
print(f"✅ Success: {len(success)}/{len(failed_brands)} ({len(success)/len(failed_brands)*100:.1f}%)")
print(f"❌ Still Failed: {len(still_failed)}/{len(failed_brands)}")

if still_failed:
    with open('still_failed_logos.json', 'w') as f:
        json.dump(still_failed, f, indent=2)
    print(f"\n💾 Still failed logos saved to still_failed_logos.json")

print(f"\n✨ Downloaded {len(success)} additional logos!")
