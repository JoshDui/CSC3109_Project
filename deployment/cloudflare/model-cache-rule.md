# Model cache rule sketch

Create a Cloudflare Cache Rule for the R2 custom model hostname.

Example match:

```text
Hostname equals "models.example.com"
AND URI Path starts with "/models/"
```

Recommended settings:

```text
Cache eligibility: Eligible for cache
Edge TTL: 1 year
Browser TTL: Respect origin or 1 year
Cache key: include host + path + query string only if signed URLs use query params
```

If using signed URLs with query params, decide whether the signature should be in
the cache key. For public immutable model paths, avoid query params and put the
version in the path instead.
