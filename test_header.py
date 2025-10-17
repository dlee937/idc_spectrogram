def parse_usrp_header(hdr_filepath):
    """Parse USRP .hdr file for metadata"""
    metadata = {}
    with open(hdr_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                try:
                    metadata[key.strip()] = float(value.strip())
                except:
                    metadata[key.strip()] = value.strip()
    return metadata

# Test on epoch_23
print("=" * 50)
print("EPOCH_23 METADATA:")
print("=" * 50)
hdr = parse_usrp_header('data/raw/epoch_23.sc16.hdr')
for key, value in hdr.items():
    try:
        if isinstance(value, (int, float)) and 'freq' in key.lower():
            print(f"{key}: {value/1e9:.3f} GHz")
        elif isinstance(value, (int, float)) and 'rate' in key.lower():
            print(f"{key}: {value/1e6:.2f} MS/s")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value}")
    except:
        pass  # Skip non-numeric metadata with encoding issues

print("\n" + "=" * 50)
print("TEST4_2412 METADATA:")
print("=" * 50)
hdr2 = parse_usrp_header('data/raw/test4_2412.sc16.hdr')
for key, value in hdr2.items():
    try:
        if isinstance(value, (int, float)) and 'freq' in key.lower():
            print(f"{key}: {value/1e9:.3f} GHz")
        elif isinstance(value, (int, float)) and 'rate' in key.lower():
            print(f"{key}: {value/1e6:.2f} MS/s")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value}")
    except:
        pass  # Skip non-numeric metadata with encoding issues
