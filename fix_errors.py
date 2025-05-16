with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Fix - Lines 586-589 if-else indentation
try:
    for i in range(len(lines)):
        if 'if is_likely_subscriber:' in lines[i]:
            # Next few lines should contain the else issue
            for j in range(i, i+10):  # Check next 10 lines
                if 'else:' in lines[j] and lines[j].startswith('                    else:'):
                    # Fix the indentation of else:
                    lines[j] = '                        else:\n'
                    # Fix the indentation of the following line too
                    if j+1 < len(lines) and 'st.info' in lines[j+1]:
                        lines[j+1] = '                            ' + lines[j+1].strip() + '\n'
                    print(f'Fixed first issue around line {j}')
                    break
            break
except Exception as e:
    print(f'Error fixing first issue: {e}')

# 2. Fix - Lines 684-687 if-elif-else indentation
try:
    for i in range(len(lines)):
        if 'def get_color(importance):' in lines[i]:
            # Look for the else: in this function
            for j in range(i, i+10):  # Check next 10 lines
                if 'else:' in lines[j] and lines[j].startswith('                    else:'):
                    # Fix the indentation of else:
                    lines[j] = '                        else:\n'
                    # Fix the indentation of the following line too
                    if j+1 < len(lines) and 'return' in lines[j+1]:
                        lines[j+1] = '                            ' + lines[j+1].strip() + '\n'
                    print(f'Fixed second issue around line {j}')
                    break
            break
except Exception as e:
    print(f'Error fixing second issue: {e}')

# Write back the fixed file
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done fixing app.py') 