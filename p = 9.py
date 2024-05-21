p = 9
q = 5
r = 8

for r in range(5, 7):
    if (p + r) < (r - p) or q < p:
        q = p + r
    if (r & 3) < p:
        p = p ^ r
        continue
    p = p + p

result = p + q + r
print(result)
