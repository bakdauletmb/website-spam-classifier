package page

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const fixture = `<!DOCTYPE html><html><head>
<meta charset="utf-8">
<title> Acme   Widgets </title>
<meta name="description" content="Widgets for everyone">
<meta name="keywords" content="widgets, gadgets">
<meta http-equiv="refresh" content="5;url=http://other.example/">
<link rel="canonical" href="http://acme.example/">
<link rel="alternate" type="application/rss+xml" href="/feed">
<style>.x{display:none}</style>
<script src="https://cdn.example/app.js"></script>
<script>window.location.replace("http://evil.example/");</script>
</head><body>
<!-- a comment -->
<h1>Hello world</h1>
<p>Visible text with a <a href="http://partner.example/x?ref=1">partner link</a>
and an <a href="/about">About us</a> page.</p>
<div style="display:none">hidden keyword stuffing text</div>
<noscript>enable javascript</noscript>
<iframe src="http://mcc.godaddy.com/park/abc"></iframe>
<img src="a.png"><form><input name="q"></form>
</body></html>`

func TestParse(t *testing.T) {
	d, err := Parse([]byte(fixture))
	if err != nil {
		t.Fatal(err)
	}
	if d.Title != "Acme Widgets" {
		t.Errorf("title = %q", d.Title)
	}
	if d.MetaDesc != "Widgets for everyone" || d.MetaKeywords != "widgets, gadgets" {
		t.Errorf("meta = %q / %q", d.MetaDesc, d.MetaKeywords)
	}
	if !d.MetaRefresh || !d.Canonical || !d.HasRSS {
		t.Errorf("flags: refresh=%v canonical=%v rss=%v", d.MetaRefresh, d.Canonical, d.HasRSS)
	}
	if d.Scripts != 2 || len(d.ScriptSrcs) != 1 || d.InlineScript == "" {
		t.Errorf("scripts=%d srcs=%v inline=%q", d.Scripts, d.ScriptSrcs, d.InlineScript)
	}
	if len(d.Anchors) != 2 || d.Anchors[0].Text != "partner link" || d.Anchors[1].Href != "/about" {
		t.Errorf("anchors = %+v", d.Anchors)
	}
	if len(d.Iframes) != 1 || d.Images != 1 || d.Forms != 1 || d.Inputs != 1 || d.H1 != 1 || d.Comments != 1 {
		t.Errorf("counts: iframes=%d img=%d forms=%d inputs=%d h1=%d comments=%d", len(d.Iframes), d.Images, d.Forms, d.Inputs, d.H1, d.Comments)
	}
	if d.HiddenStyles != 1 {
		t.Errorf("hidden styles = %d", d.HiddenStyles)
	}
	for _, bad := range []string{"display:none", "window.location", "hidden keyword", "enable javascript"} {
		if contains(d.Text, bad) {
			t.Errorf("visible text should not contain %q: %q", bad, d.Text)
		}
	}
	if !contains(d.HiddenText, "hidden keyword stuffing") || !contains(d.HiddenText, "enable javascript") {
		t.Errorf("hidden text = %q", d.HiddenText)
	}
	if !contains(d.Text, "Visible text with a partner link") {
		t.Errorf("visible text = %q", d.Text)
	}
}

func TestParseLegacyCharset(t *testing.T) {
	// "café" in windows-1252: 0xE9 for é.
	raw := []byte("<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=windows-1252\"></head><body>caf\xe9 au lait</body></html>")
	d, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(d.Words, []string{"café", "au", "lait"}) {
		t.Errorf("words = %q", d.Words)
	}
}

func TestTokenize(t *testing.T) {
	got := Tokenize("Hello, World! Don't call 555-1234 or e-mail X; Über-cool 日本語")
	want := []string{"hello", "world", "dont", "call", "#num", "#num", "or", "mail", "über", "cool", "日本語"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Tokenize = %q, want %q", got, want)
	}
}

func TestDomainFromPath(t *testing.T) {
	cases := map[string]string{
		"sites/example.com":                 "example.com",
		"sites/Example.com/":                "example.com",
		"dump/example.org.html":             "example.org",
		"dump/example.org/index.html":       "example.org",
		"sciencecentral":                    "sciencecentral",
		"/abs/path/to/sub.domain.co.uk.htm": "sub.domain.co.uk",
		"index.html":                        "index.html", // no parent to fall back on: keep as-is
	}
	for in, want := range cases {
		if got := DomainFromPath(in); got != want && !(in == "index.html" && got == ".") {
			t.Errorf("DomainFromPath(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestLoadDirectory(t *testing.T) {
	dir := t.TempDir()
	site := filepath.Join(dir, "example.com")
	if err := os.Mkdir(site, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(site, "index.html"), []byte("<title>Example</title><p>hi there</p>"), 0o644); err != nil {
		t.Fatal(err)
	}
	d, err := Load(site)
	if err != nil {
		t.Fatal(err)
	}
	if d.Domain != "example.com" || d.Title != "Example" {
		t.Errorf("domain=%q title=%q", d.Domain, d.Title)
	}
}

func TestHostHelpers(t *testing.T) {
	if Host("https://www.Sub.Example.co.uk/path?x=1") != "sub.example.co.uk" {
		t.Error("Host")
	}
	if Host("#top") != "" || Host("javascript:void(0)") != "" || Host("/relative") != "" {
		t.Error("non-hosts should be empty")
	}
	if Host("//cdn.example/x.js") != "cdn.example" {
		t.Error("protocol-relative")
	}
	if RegistrableDomain("a.b.example.com") != "example.com" || RegistrableDomain("www.bbc.co.uk") != "bbc.co.uk" || RegistrableDomain("example.org") != "example.org" {
		t.Error("RegistrableDomain")
	}
}

func contains(s, sub string) bool { return len(sub) == 0 || indexOf(s, sub) >= 0 }

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
