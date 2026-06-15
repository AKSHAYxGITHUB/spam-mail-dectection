"""Engineered spam signals (zero-dependency, standard library only).

Real-world filters such as SpamAssassin score a message by combining a machine
-learning model with a large set of weighted heuristic rules. This module
implements a compact version of that idea: each rule that fires contributes a
weight measured in *log-odds*, so the signal score can be added directly to the
ML model's log-odds before squashing back to a probability.

Positive weight  -> more spammy.  Weights are deliberately modest, so a single
signal nudges the score while several together can flip a borderline message.
"""
import re

# --- Pattern tables -------------------------------------------------------

URL_RE = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
IP_URL_RE = re.compile(r'https?://\d{1,3}(?:\.\d{1,3}){3}', re.IGNORECASE)
FROM_RE = re.compile(r'^\s*from:.*?([\w.+-]+@[\w.-]+)', re.IGNORECASE | re.MULTILINE)
HTML_RE = re.compile(r'<\s*(a|div|table|span|img|html|body|font)\b', re.IGNORECASE)

URL_SHORTENERS = {
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly',
    'rebrand.ly', 'cutt.ly', 'shorturl.at', 'rb.gy',
}
SUSPICIOUS_TLDS = (
    '.xyz', '.top', '.club', '.click', '.work', '.loan', '.win', '.gq',
    '.tk', '.ml', '.cf', '.ga', '.country', '.stream', '.download',
)
FREE_PROVIDERS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
    'mail.com', 'gmx.com', 'protonmail.com', 'yandex.com',
}
BRANDS = (
    'paypal', 'amazon', 'apple', 'microsoft', 'netflix', 'bank', 'hsbc',
    'chase', 'wells fargo', 'irs', 'fedex', 'dhl', 'ups', 'google',
)
URGENCY_PHRASES = (
    'act now', 'urgent', 'immediately', 'verify your account', 'confirm your account',
    'account suspended', 'has been suspended', 'limited time', 'expire', 'final notice',
    'last chance', 'within 24 hours', 'action required', 'update your', 'click here',
    'log in to verify', 'unusual activity', 'security alert',
)
MONEY_PHRASES = (
    'you have won', 'you won', 'winner', 'lottery', 'prize', 'cash prize',
    'million dollars', 'inheritance', 'wire transfer', 'bitcoin', 'crypto',
    'gift card', 'refund', 'claim your', 'reward', 'free money', 'investment opportunity',
)
GREED_PHRASES = (
    '100% free', 'risk free', 'risk-free', 'no cost', 'double your', 'guaranteed',
    'satisfaction guaranteed', 'best price', 'lowest price', 'cheap', 'discount',
    'special promotion', 'order now', 'buy now', 'limited offer',
)
GENERIC_GREETINGS = (
    'dear customer', 'dear friend', 'dear user', 'dear account holder',
    'dear valued customer', 'dear sir/madam', 'attention',
)

MAX_TOTAL_LOGODDS = 4.0  # cap so signals can't fully dominate the ML model


def _domain_of(email):
    return email.rsplit('@', 1)[-1].lower() if '@' in email else ''


def _count_phrases(low, phrases):
    return [p for p in phrases if p in low]


def extract_signals(text):
    """Return a list of fired signals: {label, detail, weight}."""
    if not text:
        return []
    low = text.lower()
    signals = []

    # --- Links ---------------------------------------------------------
    urls = URL_RE.findall(text)
    if IP_URL_RE.search(text):
        signals.append({'label': 'Raw IP-address link',
                        'detail': 'Links to a bare IP instead of a domain',
                        'weight': 1.6})
    if any(short in low for short in URL_SHORTENERS):
        signals.append({'label': 'URL shortener',
                        'detail': 'Hides the real destination behind a short link',
                        'weight': 1.0})
    if any(tld in low for tld in SUSPICIOUS_TLDS):
        signals.append({'label': 'Suspicious domain (TLD)',
                        'detail': 'Uses a TLD frequently abused by spammers',
                        'weight': 0.9})
    if len(urls) >= 3:
        signals.append({'label': 'Many links',
                        'detail': f'{len(urls)} links in the message',
                        'weight': 0.7})
    elif len(urls) >= 1:
        signals.append({'label': 'Contains a link',
                        'detail': f'{len(urls)} link(s) found',
                        'weight': 0.2})

    # --- Shouting / punctuation ---------------------------------------
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 20:
        caps_ratio = sum(c.isupper() for c in letters) / len(letters)
        if caps_ratio > 0.45:
            signals.append({'label': 'Mostly UPPERCASE',
                            'detail': f'{caps_ratio*100:.0f}% of letters are capitals',
                            'weight': 1.0})
        elif caps_ratio > 0.25:
            signals.append({'label': 'Excessive capitals',
                            'detail': f'{caps_ratio*100:.0f}% of letters are capitals',
                            'weight': 0.4})
    if text.count('!') >= 3:
        signals.append({'label': 'Excessive exclamation marks',
                        'detail': f"{text.count('!')} exclamation marks",
                        'weight': 0.6})

    # --- Scammy language ----------------------------------------------
    urgency = _count_phrases(low, URGENCY_PHRASES)
    if urgency:
        signals.append({'label': 'Urgency / pressure language',
                        'detail': ', '.join(urgency[:3]),
                        'weight': min(0.5 * len(urgency), 1.5)})
    money = _count_phrases(low, MONEY_PHRASES)
    if money:
        signals.append({'label': 'Money / prize lure',
                        'detail': ', '.join(money[:3]),
                        'weight': min(0.4 * len(money), 1.2)})
    greed = _count_phrases(low, GREED_PHRASES)
    if greed:
        signals.append({'label': 'Sales / "too good to be true"',
                        'detail': ', '.join(greed[:3]),
                        'weight': min(0.3 * len(greed), 0.9)})
    generic = _count_phrases(low, GENERIC_GREETINGS)
    if generic:
        signals.append({'label': 'Generic greeting',
                        'detail': generic[0],
                        'weight': 0.4})

    # --- Formatting ----------------------------------------------------
    if HTML_RE.search(text):
        signals.append({'label': 'HTML markup',
                        'detail': 'Message contains HTML tags',
                        'weight': 0.3})

    # --- Sender / spoofing (only if a From: header was pasted) ---------
    m = FROM_RE.search(text)
    if m:
        domain = _domain_of(m.group(1))
        claims_brand = next((b for b in BRANDS if b in low), None)
        if claims_brand and domain in FREE_PROVIDERS:
            signals.append({'label': 'Possible sender spoofing',
                            'detail': f'Claims to be "{claims_brand}" but sent from {domain}',
                            'weight': 1.8})

    return signals


def total_logodds(signals):
    """Sum of signal weights, capped so signals cannot fully override the model."""
    return min(sum(s['weight'] for s in signals), MAX_TOTAL_LOGODDS)


if __name__ == '__main__':
    sample = "URGENT!! Dear Customer, your PayPal account has been suspended. Verify now: http://bit.ly/x"
    for s in extract_signals(sample):
        print(f"+{s['weight']:.2f}  {s['label']} - {s['detail']}")
