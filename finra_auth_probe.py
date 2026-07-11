#!/usr/bin/env python3
"""
FINRA OAUTH TOKEN PROBE — diagnoses the 400 on token exchange.
Tries the token request several documented ways and prints the EXACT response
body so we can see what FINRA objects to. Read-only, one tiny call per variant.

USAGE:
  python finra_auth_probe.py --client-id ID --client-secret SECRET
"""
import argparse, base64, json, sys
import urllib.request, urllib.error, urllib.parse

OAUTH="https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"

def attempt(label, req):
    print("\n--- %s ---"%label)
    print("  URL:", req.full_url)
    print("  method:", req.get_method())
    print("  headers:", {k:(v[:24]+"..." if k.lower()=="authorization" else v) for k,v in req.headers.items()})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code=r.getcode(); body=r.read().decode("utf-8","replace")
            print("  -> HTTP %s"%code)
            print("  body:", body[:300])
            try:
                tok=json.loads(body).get("access_token")
                if tok: print("  ** SUCCESS — access_token received (…%s) **"%tok[-6:]); return True
            except Exception: pass
    except urllib.error.HTTPError as e:
        body=e.read().decode("utf-8","replace")
        print("  -> HTTP %s"%e.code)
        print("  body:", body[:400])
    except Exception as e:
        print("  -> ERROR:", str(e)[:200])
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--client-id",required=True)
    ap.add_argument("--client-secret",required=True)
    a=ap.parse_args()
    cid=a.client_id.strip(); sec=a.client_secret.strip()
    cred=base64.b64encode(("%s:%s"%(cid,sec)).encode()).decode()
    print("FINRA OAuth probe")
    print("  client_id: %s (len %d)"%(cid,len(cid)))
    print("  secret: …%s (len %d)"%(sec[-4:],len(sec)))
    print("  base64(id:secret): %s…"%cred[:24])

    ok=False

    # Variant 1: grant_type in QUERY string, empty body, Basic header (FINRA doc canonical)
    r=urllib.request.Request(OAUTH+"?grant_type=client_credentials", data=b"", method="POST",
        headers={"Authorization":"Basic "+cred})
    ok=attempt("V1: grant in query, empty body, Basic header", r) or ok

    # Variant 2: grant_type in BODY (form-encoded), Basic header
    body=urllib.parse.urlencode({"grant_type":"client_credentials"}).encode()
    r=urllib.request.Request(OAUTH, data=body, method="POST",
        headers={"Authorization":"Basic "+cred,"Content-Type":"application/x-www-form-urlencoded"})
    ok=attempt("V2: grant in form body, Basic header", r) or ok

    # Variant 3: credentials in BODY (no Basic header), grant in body
    body=urllib.parse.urlencode({"grant_type":"client_credentials","client_id":cid,"client_secret":sec}).encode()
    r=urllib.request.Request(OAUTH, data=body, method="POST",
        headers={"Content-Type":"application/x-www-form-urlencoded"})
    ok=attempt("V3: client_id/secret in body, no Basic header", r) or ok

    # Variant 4: query grant + Basic header + explicit Content-Type + Accept
    r=urllib.request.Request(OAUTH+"?grant_type=client_credentials", data=b"", method="POST",
        headers={"Authorization":"Basic "+cred,
                 "Content-Type":"application/x-www-form-urlencoded","Accept":"application/json"})
    ok=attempt("V4: query grant + Basic + Content-Type + Accept", r) or ok

    print("\n==============================================================")
    if ok:
        print("At least one variant SUCCEEDED — note which, and I'll lock the fetcher to it.")
    else:
        print("All variants failed. Paste this whole output. The response body usually says WHY")
        print("(e.g. 'invalid_client' = wrong creds; 'unsupported_grant_type' = format; a")
        print("Cloudflare/HTML page = the call is being blocked before reaching the OAuth service).")
        print("Also double-check: did you paste the SECRET exactly (no trailing space), and is the")
        print("credential's Type set to 'Public' and ACTIVE in the FINRA console?")

if __name__=="__main__":
    main()
