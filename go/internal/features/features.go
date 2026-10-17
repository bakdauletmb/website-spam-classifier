// Package features turns a parsed page.Document into a sparse feature
// vector. Two families of features are produced:
//
//   - "signals": hand-written, human-readable indicators in the spirit of
//     SpamAssassin rules (parked-domain templates, redirect stubs, keyword
//     stuffing, link farms, missing site structure, domain shape, ...).
//     Numeric quantities are bucketed into indicator features so the linear
//     model can learn non-monotonic effects without any scaling.
//   - "tokens": bag-of-words presence features for the title, meta tags,
//     anchor text, body text, the domain name and the hosts the page links
//     to or embeds.
//
// All feature values lie in [0,1], which keeps logistic regression training
// well-conditioned without per-feature standardization.
package features

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"spamclassifier/internal/page"
)

// Mode selects which feature families to emit.
type Mode string

const (
	ModeAll     Mode = "all"
	ModeSignals Mode = "signals"
	ModeTokens  Mode = "tokens"
)

// ParseMode validates a user-supplied mode string.
func ParseMode(s string) (Mode, error) {
	switch Mode(s) {
	case ModeAll, ModeSignals, ModeTokens:
		return Mode(s), nil
	}
	return "", fmt.Errorf("unknown feature mode %q (want all|signals|tokens)", s)
}

// Vector is a sparse feature vector.
type Vector map[string]float64

// Names returns the feature names in sorted order (for deterministic output).
func (v Vector) Names() []string {
	names := make([]string, 0, len(v))
	for k := range v {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

func (v Vector) set(name string) { v[name] = 1 }

// bucket emits one indicator for the bucket that val falls into, e.g.
// "sig:words<=10", "sig:words<=50", ..., "sig:words>5000".
func (v Vector) bucket(name string, val int, edges []int) {
	for _, e := range edges {
		if val <= e {
			v[fmt.Sprintf("sig:%s<=%d", name, e)] = 1
			return
		}
	}
	v[fmt.Sprintf("sig:%s>%d", name, edges[len(edges)-1])] = 1
}

// ratio emits a clipped continuous feature in [0,1].
func (v Vector) ratio(name string, val float64) {
	if math.IsNaN(val) || val < 0 {
		val = 0
	}
	if val > 1 {
		val = 1
	}
	if val > 0 {
		v["sig:"+name] = val
	}
}

const (
	maxBodyTokens     = 3000 // tokens of body text considered for features
	maxAnchorsSeen    = 400
	maxHosts          = 60
	templateScanBytes = 20000
)

var (
	// Parking / for-sale templates. These pages exist only to serve ads.
	reParked = regexp.MustCompile(`(this (web ?site|domain|page) is for sale|buy this domain|domain (name )?(is|may be) for sale|is for sale!|sponsored listings|related searches|parked (free|domain)|domain parking|parkingcrew|sedoparking|sedo\.com|hugedomains|afternic|dan\.com|uniregistry|bodis\.com|above\.com|dsnextgen|parklogic|skenzo|namedrive|domainsponsor|godaddy\.com/park|searchfusion|is coming soon|website coming soon|under construction|this domain has expired|domain has expired|renew now|inquire about this domain|make an offer)`)
	// Hosting error / placeholder pages.
	reErrorPage = regexp.MustCompile(`(page cannot be displayed|account (has been )?suspended|403 forbidden|404 not found|not found on this server|default web site page|apache2? .* default page|it works!|welcome to nginx|index of /|bandwidth limit exceeded|site temporarily unavailable|error establishing a database connection|service unavailable|internal server error|directory listing)`)
	// JavaScript redirect stubs.
	reJSRedirect = regexp.MustCompile(`(location\.(replace|href|assign)|window\.location|document\.location|top\.location|self\.location|meta.*refresh)`)
	// Ad-serving and traffic-arbitrage hosts.
	reAdHost = regexp.MustCompile(`(googlesyndication|doubleclick|adsbygoogle|google_ad_client|adbrite|adsonar|adroll|zedo\.com|clicksor|infolinks|chitika|bidvertiser|adhitz|exoclick|juicyads|trafficjunky|propellerads|popads|popcash|revenuehits|adcash|kontera|outbrain|taboola|mgid|yllix|adf\.ly|linkbucks|sh\.st)`)
	// Affiliate link patterns.
	reAffiliate = regexp.MustCompile(`(amazon\.[a-z.]+/.*[?&]tag=|clickbank|\.hop\.|shareasale|cj\.com|anrdoezrs|dpbolvw|jdoqocy|tkqlhce|awin1|linksynergy|avantlink|pepperjam|impactradius|affiliate|partner_?id=|aff_?id=|ref=|utm_source=)`)
	// Structural markers of a maintained web site.
	reSiteNav   = regexp.MustCompile(`^(about( us)?|contact( us)?|privacy( policy)?|terms( of (use|service))?|login|log in|sign in|sign up|register|faq|help|support|careers|jobs|blog|news|home|search|sitemap|rss|subscribe|cart|checkout|my account|press|team|pricing|docs|documentation|download|products|services|portfolio|gallery|events|forum|community|advertise|legal|disclaimer|accessibility|impressum|datenschutz|kontakt|über uns)$`)
	reCopyright = regexp.MustCompile(`(©|&copy;|copyright|all rights reserved|\(c\) ?(19|20)\d\d)`)
	reNavMarker = regexp.MustCompile(`\b(privacy policy|terms of (use|service)|contact us|about us|sitemap|site map|all rights reserved)\b`)
	reTLDBad    = regexp.MustCompile(`^(info|biz|us|cn|ru|tk|ws|cc|mobi|xyz|top|club|online|site|loan|work|click|link|date|racing|download|stream|win|bid|trade|party|review|science|gq|ml|cf|ga)$`)

	stopWords = toSet(strings.Fields(`a an and are as at be by for from has he in is it its of on or that the to was were will with this these those you your we our they their i me my not but if then than so do does did have had can could would should may might shall all any each more most other some such no nor only own same too very s t just don now also about into over after before under again further here there when where why how what which who whom`))
)

// spamVocab lists unigrams and bigrams that are heavily over-represented in
// web spam (parked pages, pharma/casino/loan affiliates, warez, link farms).
// Matching is done on the token stream, which is far cheaper than a large
// case-insensitive regular expression over the raw text.
var spamVocab = toSet(strings.Split(`free download|download free|free online|100 free|casino|poker|slots|viagra|cialis|pharmacy|pills|payday loan|payday loans|cash advance|debt consolidation|credit repair|work from home|make money|earn money|get rich|weight loss|diet pills|cheap|discount|best price|lowest price|torrent|torrents|warez|crack|cracks|keygen|serial key|rapidshare|megaupload|hotfile|ringtones|adult|xxx|porn|sex|dating|singles|mortgage rates|insurance quotes|refinance|forex|binary options|seo services|link building|backlinks|buy now|click here|limited time|act now|free trial|no obligation|risk free|guaranteed|replica|rolex|louis vuitton|handbags|ugg boots|nike shoes|jerseys|essay writing|term papers|celebrity|nude|lyrics|wallpapers|cheats|hacks|generator|unblocked|proxy|anonymous surfing|vpn|hosting|domain names|web directory|link exchange|add url|submit url|reciprocal link|sponsored results|sponsored listings|search results|related searches|for sale|buy this domain|this domain|domain is|domain name|domain names|web site is|site is for|is for sale|find results|results fast|privacy policy terms|sponsored|listings|parked|advertisers|advertise here|click to continue|continue to|proceed|redirecting|loading`, "|"))

// countSpamVocab counts unigram and bigram hits in a token stream.
func countSpamVocab(words []string) int {
	n := 0
	for i, w := range words {
		if spamVocab[w] {
			n++
		}
		if i+1 < len(words) && spamVocab[w+" "+words[i+1]] {
			n++
		}
	}
	return n
}

func toSet(words []string) map[string]bool {
	m := make(map[string]bool, len(words))
	for _, w := range words {
		m[w] = true
	}
	return m
}

// Extract computes the feature vector for d.
func Extract(d *page.Document, mode Mode) Vector {
	v := Vector{}
	if mode == ModeAll || mode == ModeSignals {
		signals(d, v)
	}
	if mode == ModeAll || mode == ModeTokens {
		tokens(d, v)
	}
	return v
}

func signals(d *page.Document, v Vector) {
	words := len(d.Words)
	links := len(d.Anchors)
	textBytes := len(d.Text)

	// --- Size and shape -------------------------------------------------
	v.bucket("bytes", d.RawBytes, []int{300, 1000, 3000, 10000, 30000, 100000, 300000})
	v.bucket("words", words, []int{0, 5, 20, 60, 150, 400, 1000, 3000})
	v.bucket("links", links, []int{0, 3, 10, 30, 80, 200, 500})
	v.bucket("scripts", d.Scripts, []int{0, 1, 3, 8, 20})
	v.bucket("iframes", len(d.Iframes), []int{0, 1, 3})
	v.bucket("images", d.Images, []int{0, 2, 10, 40})
	v.bucket("forms", d.Forms, []int{0, 1, 3})
	v.bucket("h1", d.H1, []int{0, 1, 3})
	v.bucket("headings", d.Headings, []int{0, 2, 6, 20})
	v.bucket("paragraphs", d.Paragraphs, []int{0, 2, 8, 30})
	v.bucket("hidden", d.HiddenStyles, []int{0, 2, 6, 20})
	v.bucket("comments", d.Comments, []int{0, 3, 15})
	if d.RawBytes > 0 {
		v.ratio("text_ratio", float64(textBytes)/float64(d.RawBytes))
	}
	v.ratio("links_per_word", float64(links)/float64(words+10))
	if words > 0 {
		v.ratio("nonascii_ratio", float64(d.NonASCII)/float64(words*5+1))
	}

	// --- Title / meta --------------------------------------------------
	title := strings.ToLower(d.Title)
	switch {
	case title == "":
		v.set("sig:no_title")
	case normalizeHost(title) == normalizeHost(d.Domain) || title == d.Domain || strings.TrimPrefix(title, "www.") == d.Domain:
		v.set("sig:title_is_domain")
	case strings.Contains(title, d.Domain):
		v.set("sig:title_contains_domain")
	}
	if len(d.Title) > 70 {
		v.set("sig:title_long")
	}
	if d.MetaDesc == "" {
		v.set("sig:no_meta_desc")
	}
	kw := page.Tokenize(d.MetaKeywords)
	v.bucket("meta_keywords", len(kw), []int{0, 5, 15, 40})
	if d.MetaRefresh {
		v.set("sig:meta_refresh")
	}
	if d.Canonical {
		v.set("sig:has_canonical")
	}
	if d.HasViewport {
		v.set("sig:has_viewport")
	}
	if d.HasOpenGraph {
		v.set("sig:has_opengraph")
	}
	if d.HasRSS {
		v.set("sig:has_rss")
	}

	// --- Templates: parked, error, redirect ------------------------------
	// Template detection only needs the head of the page: parked, error and
	// redirect pages are tiny, and scanning 300 KB of news text for them is
	// wasted work.
	lowerAll := head(strings.ToLower(d.Title+" "+d.MetaDesc+" "+d.Text+" "+d.HiddenText), templateScanBytes)
	lowerVisible := head(strings.ToLower(d.Title+" "+d.Text), templateScanBytes)
	if reParked.MatchString(lowerAll) || anyMatch(d.Iframes, reParked) || anyMatch(d.ScriptSrcs, reParked) {
		v.set("sig:parked_template")
	}
	if reErrorPage.MatchString(lowerVisible) {
		v.set("sig:error_page")
	}
	if reJSRedirect.MatchString(strings.ToLower(d.InlineScript)) {
		v.set("sig:js_redirect")
	}
	if words < 5 && d.Scripts > 0 && links == 0 {
		v.set("sig:script_only_stub")
	}
	if words < 15 && len(d.Iframes) > 0 {
		v.set("sig:iframe_only_stub")
	}
	if words < 15 && links == 1 {
		v.set("sig:single_link_stub")
	}
	if d.HiddenText != "" {
		hw := len(page.Tokenize(d.HiddenText))
		v.bucket("hidden_words", hw, []int{0, 10, 50, 200})
		if hw > words && hw > 20 {
			v.set("sig:more_hidden_than_visible")
		}
	}

	// --- Keyword stuffing / lexical diversity -----------------------------
	if words >= 20 {
		uniq := map[string]int{}
		top, topWord := 0, ""
		for _, w := range d.Words {
			uniq[w]++
			if !stopWords[w] && uniq[w] > top {
				top, topWord = uniq[w], w
			}
		}
		_ = topWord
		div := float64(len(uniq)) / float64(words)
		v.ratio("lexical_diversity", div)
		if div < 0.25 {
			v.set("sig:low_lexical_diversity")
		}
		share := float64(top) / float64(words)
		v.ratio("top_word_share", share*4) // scale so 25% share saturates
		if share > 0.06 {
			v.set("sig:keyword_repeated")
		}
	}
	spamHits := countSpamVocab(d.Words) + countSpamVocab(page.Tokenize(d.Title+" "+d.MetaDesc+" "+d.MetaKeywords)) + countSpamVocab(page.Tokenize(d.HiddenText))
	v.bucket("spam_vocab", spamHits, []int{0, 1, 3, 8, 20})
	if words > 0 {
		v.ratio("spam_vocab_density", float64(spamHits)*20/float64(words))
	}

	// --- Link profile ------------------------------------------------------
	var ext, internal, empty, navLike, affiliate, longAnchor int
	hosts := map[string]int{}
	for i, a := range d.Anchors {
		if i >= maxAnchorsSeen*4 {
			break
		}
		h := page.Host(a.Href)
		switch {
		case a.Href == "" || strings.HasPrefix(a.Href, "#") || strings.HasPrefix(a.Href, "javascript:"):
			empty++
		case h == "" || sameSite(h, d.Domain):
			internal++
		default:
			ext++
			hosts[page.RegistrableDomain(h)]++
		}
		t := strings.ToLower(strings.TrimSpace(a.Text))
		if reSiteNav.MatchString(t) {
			navLike++
		}
		if len(strings.Fields(t)) >= 6 {
			longAnchor++
		}
		if reAffiliate.MatchString(strings.ToLower(a.Href)) {
			affiliate++
		}
	}
	if links > 0 {
		v.ratio("ext_link_ratio", float64(ext)/float64(links))
		v.ratio("empty_link_ratio", float64(empty)/float64(links))
		v.ratio("long_anchor_ratio", float64(longAnchor)/float64(links))
	}
	v.bucket("ext_links", ext, []int{0, 2, 10, 40, 150})
	v.bucket("ext_hosts", len(hosts), []int{0, 2, 8, 25, 80})
	v.bucket("nav_links", navLike, []int{0, 1, 3, 6})
	v.bucket("affiliate_links", affiliate, []int{0, 1, 5})
	if links > 30 && ext > 0 && float64(ext)/float64(links) > 0.8 {
		v.set("sig:link_farm_shape")
	}
	if words > 100 && float64(links)/float64(words) > 0.4 {
		v.set("sig:link_dense_text")
	}

	// --- Ads and monetization --------------------------------------------
	adHits := 0
	for _, s := range d.ScriptSrcs {
		if reAdHost.MatchString(strings.ToLower(s)) {
			adHits++
		}
	}
	for _, s := range d.Iframes {
		if reAdHost.MatchString(strings.ToLower(s)) {
			adHits++
		}
	}
	if reAdHost.MatchString(strings.ToLower(d.InlineScript)) {
		adHits++
	}
	v.bucket("ad_hits", adHits, []int{0, 1, 3})
	if adHits > 0 && words < 150 {
		v.set("sig:ads_with_thin_content")
	}

	// --- Signs of a maintained site ----------------------------------------
	if reCopyright.MatchString(lowerVisible) {
		v.set("sig:has_copyright")
	}
	if reNavMarker.MatchString(lowerVisible) {
		v.set("sig:has_nav_markers")
	}
	if d.Inputs > 0 {
		v.set("sig:has_inputs")
	}

	// --- Domain shape --------------------------------------------------------
	dom := d.Domain
	labels := strings.Split(dom, ".")
	tld := labels[len(labels)-1]
	if len(labels) > 1 {
		v.set("sig:tld:" + tld)
		if reTLDBad.MatchString(tld) {
			v.set("sig:tld_risky")
		}
	}
	name := strings.Join(labels[:max(1, len(labels)-1)], ".")
	v.bucket("domain_len", len(name), []int{6, 10, 15, 22})
	v.bucket("domain_hyphens", strings.Count(name, "-"), []int{0, 1, 2})
	if strings.ContainsAny(name, "0123456789") {
		v.set("sig:domain_has_digit")
	}
	domTokens := splitDomainTokens(name)
	if len(domTokens) >= 3 {
		v.set("sig:domain_many_tokens")
	}
	if countSpamVocab(domTokens) > 0 {
		v.set("sig:domain_spam_vocab")
	}
}

func tokens(d *page.Document, v Vector) {
	seen := map[string]bool{}
	add := func(prefix, tok string) {
		if tok == "" || stopWords[tok] {
			return
		}
		k := prefix + tok
		if !seen[k] {
			seen[k] = true
			v[k] = 1
		}
	}
	for i, w := range d.Words {
		if i >= maxBodyTokens {
			break
		}
		add("w:", w)
	}
	for _, w := range page.Tokenize(d.Title) {
		add("ti:", w)
	}
	for _, w := range page.Tokenize(d.MetaDesc + " " + d.MetaKeywords) {
		add("me:", w)
	}
	for _, w := range page.Tokenize(d.HiddenText) {
		add("hd:", w)
	}
	for i, a := range d.Anchors {
		if i >= maxAnchorsSeen {
			break
		}
		for _, w := range page.Tokenize(a.Text) {
			add("an:", w)
		}
	}
	// Hosts the page links to, embeds, or loads scripts from.
	n := 0
	for _, a := range d.Anchors {
		if h := page.Host(a.Href); h != "" && !sameSite(h, d.Domain) {
			add("lh:", page.RegistrableDomain(h))
			if n++; n > maxHosts {
				break
			}
		}
	}
	for _, s := range d.Iframes {
		if h := page.Host(s); h != "" {
			add("ih:", page.RegistrableDomain(h))
		}
	}
	for _, s := range d.ScriptSrcs {
		if h := page.Host(s); h != "" && !sameSite(h, d.Domain) {
			add("sh:", page.RegistrableDomain(h))
		}
	}
	// Domain name tokens, e.g. "payday-loan-info.com" -> payday, loan, info.
	labels := strings.Split(d.Domain, ".")
	for _, t := range splitDomainTokens(strings.Join(labels[:max(1, len(labels)-1)], ".")) {
		add("dom:", t)
	}
}

func head(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func anyMatch(ss []string, re *regexp.Regexp) bool {
	for _, s := range ss {
		if re.MatchString(strings.ToLower(s)) {
			return true
		}
	}
	return false
}

func normalizeHost(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimPrefix(s, "http://")
	s = strings.TrimPrefix(s, "https://")
	s = strings.TrimPrefix(s, "www.")
	return strings.TrimSuffix(s, "/")
}

func sameSite(host, domain string) bool {
	host = strings.TrimPrefix(host, "www.")
	return host == domain || strings.HasSuffix(host, "."+domain) || strings.HasSuffix(domain, "."+host)
}

// splitDomainTokens breaks a domain label into word tokens on hyphens,
// digits and dots. (A dictionary-based segmenter would do better on
// "paydayloaninfl" but is out of scope for the prototype.)
func splitDomainTokens(name string) []string {
	var out []string
	for _, t := range strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '.' || r == '_' || (r >= '0' && r <= '9') }) {
		if len(t) >= 2 {
			out = append(out, t)
		}
	}
	return out
}
