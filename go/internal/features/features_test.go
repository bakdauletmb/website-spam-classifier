package features

import (
	"strings"
	"testing"

	"spamclassifier/internal/page"
)

func parse(t *testing.T, domain, html string) *page.Document {
	t.Helper()
	d, err := page.Parse([]byte(html))
	if err != nil {
		t.Fatal(err)
	}
	d.Domain = domain
	return d
}

func TestParkedPageSignals(t *testing.T) {
	d := parse(t, "0-0.info", `<!DOCTYPE html><body style="padding:0; margin:0;"><iframe src="http://mcc.godaddy.com/park/ZP0jYaMup2V=" width="100%"></iframe></body></html>`)
	v := Extract(d, ModeAll)
	for _, want := range []string{"sig:parked_template", "sig:iframe_only_stub", "sig:no_title", "sig:words<=0", "ih:godaddy.com", "sig:tld:info", "sig:tld_risky"} {
		if v[want] != 1 {
			t.Errorf("missing %s in %v", want, v.Names())
		}
	}
	if _, ok := v["sig:has_copyright"]; ok {
		t.Error("parked page should not have copyright signal")
	}
}

func TestForSaleTemplate(t *testing.T) {
	d := parse(t, "mcsbk.com", `<title>HugeDomains.com - Mcsbk.com is for Sale (Mcsbk)</title><body>Questions? We're here to help! Buy this domain.</body>`)
	v := Extract(d, ModeSignals)
	if v["sig:parked_template"] != 1 {
		t.Error("expected parked_template")
	}
	if v["sig:title_contains_domain"] != 1 {
		t.Error("expected title_contains_domain")
	}
	if _, ok := v["w:domain"]; ok {
		t.Error("signals mode must not emit token features")
	}
}

func TestRedirectStub(t *testing.T) {
	d := parse(t, "blanketloan.com", `<html><head><script>window.location.href="http://x.example/";</script></head><body></body></html>`)
	v := Extract(d, ModeSignals)
	if v["sig:js_redirect"] != 1 || v["sig:script_only_stub"] != 1 {
		t.Errorf("redirect stub signals missing: %v", v.Names())
	}
}

func TestKeywordStuffing(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("<title>cabletvv.org</title><body>")
	for i := 0; i < 60; i++ {
		sb.WriteString("cable tv cheap cable tv service ")
	}
	sb.WriteString("</body>")
	d := parse(t, "serviceelctric.com", sb.String())
	v := Extract(d, ModeAll)
	if v["sig:low_lexical_diversity"] != 1 || v["sig:keyword_repeated"] != 1 {
		t.Errorf("stuffing signals missing: %v", v.Names())
	}
	if v["sig:spam_vocab>20"] != 1 {
		t.Errorf("expected heavy spam vocabulary, got %v", v.Names())
	}
	if v["w:cable"] != 1 || v["ti:cabletvv"] != 1 {
		t.Error("token features missing")
	}
}

func TestLegitimateSiteSignals(t *testing.T) {
	d := parse(t, "jdbank.com", `<html><head><title>JD Bank | Lake Charles</title><meta name="description" content="Community bank"><meta name="viewport" content="width=device-width"><link rel="canonical" href="https://jdbank.com/"></head>
<body><nav><a href="/about">About Us</a> <a href="/contact">Contact Us</a> <a href="/privacy">Privacy Policy</a> <a href="/login">Login</a></nav>
<h1>Welcome</h1><p>We have served Louisiana families since 1947 with checking, savings and mortgage products.</p>
<form><input name="q"></form><footer>© 2024 JD Bank. All rights reserved.</footer></body></html>`)
	v := Extract(d, ModeSignals)
	for _, want := range []string{"sig:has_copyright", "sig:has_nav_markers", "sig:has_canonical", "sig:has_viewport", "sig:has_inputs", "sig:nav_links<=6"} {
		if v[want] != 1 {
			t.Errorf("missing %s in %v", want, v.Names())
		}
	}
	for _, bad := range []string{"sig:parked_template", "sig:error_page", "sig:js_redirect", "sig:no_title"} {
		if _, ok := v[bad]; ok {
			t.Errorf("unexpected %s", bad)
		}
	}
}

func TestLinkFarmShape(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("<body>")
	for i := 0; i < 50; i++ {
		sb.WriteString(`<a href="http://site` + string(rune('a'+i%26)) + `.example/">buy cheap stuff here now</a> `)
	}
	sb.WriteString("</body>")
	d := parse(t, "rawtag.com", sb.String())
	v := Extract(d, ModeAll)
	if v["sig:link_farm_shape"] != 1 {
		t.Errorf("expected link_farm_shape: %v", v.Names())
	}
	if v["sig:ext_link_ratio"] != 1 {
		t.Errorf("ext_link_ratio = %v", v["sig:ext_link_ratio"])
	}
}

func TestValuesAreBounded(t *testing.T) {
	d := parse(t, "x.com", `<title>t</title><body>`+strings.Repeat("word ", 500)+`<a href="http://a.example">a</a></body>`)
	for name, val := range Extract(d, ModeAll) {
		if val < 0 || val > 1 {
			t.Errorf("%s = %v outside [0,1]", name, val)
		}
	}
}

func TestParseMode(t *testing.T) {
	if _, err := ParseMode("bogus"); err == nil {
		t.Error("expected error for bogus mode")
	}
	if m, _ := ParseMode("tokens"); m != ModeTokens {
		t.Error("tokens mode")
	}
}
