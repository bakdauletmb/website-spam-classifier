// Package page loads a scraped web site from disk and turns it into a
// de-noised Document that the feature extractor can reason about.
package page

import (
	"bytes"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"
	"golang.org/x/net/html/charset"
)

// MaxBytes caps how much of a page we parse. Typical home pages are well
// under this; the cap protects the classifier from pathological input.
const MaxBytes = 4 << 20

// Anchor is one <a href> element.
type Anchor struct {
	Href string
	Text string
}

// Document is the parsed view of one scraped web site.
type Document struct {
	Domain   string // detected from the file/directory name
	RawBytes int    // size of the HTML on disk

	Title        string
	MetaDesc     string
	MetaKeywords string
	MetaRefresh  bool // <meta http-equiv="refresh">
	Canonical    bool // <link rel="canonical">
	HasViewport  bool
	HasOpenGraph bool
	HasRSS       bool

	Text  string   // visible text, whitespace-normalized
	Words []string // lowercased word tokens from Text

	Anchors      []Anchor
	Scripts      int      // all <script> elements
	ScriptSrcs   []string // external script URLs
	InlineScript string   // concatenated inline script bodies
	Iframes      []string // <iframe>/<frame> src values
	Images       int
	Forms        int
	Inputs       int
	H1           int
	Headings     int // h1..h6
	Paragraphs   int
	ListItems    int
	Tables       int
	HiddenStyles int    // inline styles that hide content
	HiddenText   string // text inside hidden elements and <noscript>
	Comments     int
	NonASCII     int // non-ASCII runes in visible text
}

// Load reads a scraped site at path (a file, or a directory containing
// index.html or similar) and parses it.
func Load(path string) (*Document, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	file := path
	domain := DomainFromPath(path)
	if info.IsDir() {
		file, err = pickFile(path)
		if err != nil {
			return nil, err
		}
	}
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	raw, err := io.ReadAll(io.LimitReader(f, MaxBytes))
	if err != nil {
		return nil, err
	}
	doc, err := Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", file, err)
	}
	doc.Domain = domain
	return doc, nil
}

// DomainFromPath derives the site's host name from the path it was saved at.
// A scraped site is expected to be stored as a file named after its host,
// e.g. "sites/example.com"; we also tolerate "example.com.html" and
// "example.com/index.html".
func DomainFromPath(path string) string {
	base := filepath.Base(filepath.Clean(path))
	lower := strings.ToLower(base)
	if lower == "index.html" || lower == "index.htm" || lower == "default.htm" || lower == "default.html" {
		base = filepath.Base(filepath.Dir(filepath.Clean(path)))
		lower = strings.ToLower(base)
	}
	for _, ext := range []string{".html", ".htm"} {
		if strings.HasSuffix(lower, ext) && strings.Count(lower, ".") > 1 {
			base = base[:len(base)-len(ext)]
			break
		}
	}
	return strings.ToLower(base)
}

func pickFile(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var names []string
	for _, e := range entries {
		if e.Type().IsRegular() {
			names = append(names, e.Name())
		}
	}
	if len(names) == 0 {
		return "", fmt.Errorf("%s: directory contains no files", dir)
	}
	for _, pref := range []string{"index.html", "index.htm", "default.htm", "default.html"} {
		for _, n := range names {
			if strings.EqualFold(n, pref) {
				return filepath.Join(dir, n), nil
			}
		}
	}
	sort.Strings(names)
	return filepath.Join(dir, names[0]), nil
}

// Parse builds a Document from raw HTML bytes. The character set is sniffed
// from a BOM, a <meta charset>, or falls back to windows-1252, so legacy
// pages still tokenize into sensible words.
func Parse(raw []byte) (*Document, error) {
	d := &Document{RawBytes: len(raw)}
	r, err := charset.NewReader(bytes.NewReader(raw), "")
	if err != nil {
		r = bytes.NewReader(raw)
	}
	root, err := html.Parse(r)
	if err != nil {
		return nil, err
	}
	var text, hidden strings.Builder
	walk(root, d, &text, &hidden, false)
	d.Text = normalizeSpace(text.String())
	d.HiddenText = normalizeSpace(hidden.String())
	d.Words = Tokenize(d.Text)
	for _, r := range d.Text {
		if r > unicode.MaxASCII {
			d.NonASCII++
		}
	}
	return d, nil
}

func walk(n *html.Node, d *Document, text, hidden *strings.Builder, inHidden bool) {
	switch n.Type {
	case html.CommentNode:
		d.Comments++
		return
	case html.TextNode:
		if inHidden {
			hidden.WriteString(n.Data)
			hidden.WriteByte(' ')
		} else {
			text.WriteString(n.Data)
			text.WriteByte(' ')
		}
		return
	case html.ElementNode:
		// fallthrough to element handling below
	default:
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c, d, text, hidden, inHidden)
		}
		return
	}

	switch n.DataAtom {
	case atom.Script:
		d.Scripts++
		if src := attr(n, "src"); src != "" {
			d.ScriptSrcs = append(d.ScriptSrcs, src)
		} else if n.FirstChild != nil {
			d.InlineScript += n.FirstChild.Data + "\n"
		}
		return
	case atom.Style, atom.Template:
		return
	case atom.Noscript:
		inHidden = true
	case atom.Title:
		if d.Title == "" && n.FirstChild != nil {
			d.Title = normalizeSpace(n.FirstChild.Data)
		}
		return
	case atom.Meta:
		name := strings.ToLower(attr(n, "name"))
		prop := strings.ToLower(attr(n, "property"))
		equiv := strings.ToLower(attr(n, "http-equiv"))
		content := attr(n, "content")
		switch {
		case name == "description":
			d.MetaDesc = normalizeSpace(content)
		case name == "keywords":
			d.MetaKeywords = normalizeSpace(content)
		case name == "viewport":
			d.HasViewport = true
		case equiv == "refresh":
			d.MetaRefresh = true
		case strings.HasPrefix(prop, "og:"):
			d.HasOpenGraph = true
		}
		return
	case atom.Link:
		rel := strings.ToLower(attr(n, "rel"))
		typ := strings.ToLower(attr(n, "type"))
		if strings.Contains(rel, "canonical") {
			d.Canonical = true
		}
		if strings.Contains(rel, "alternate") && (strings.Contains(typ, "rss") || strings.Contains(typ, "atom")) {
			d.HasRSS = true
		}
		return
	case atom.A:
		var sb strings.Builder
		collectText(n, &sb)
		d.Anchors = append(d.Anchors, Anchor{Href: strings.TrimSpace(attr(n, "href")), Text: normalizeSpace(sb.String())})
	case atom.Iframe, atom.Frame:
		d.Iframes = append(d.Iframes, attr(n, "src"))
	case atom.Img:
		d.Images++
	case atom.Form:
		d.Forms++
	case atom.Input, atom.Textarea, atom.Select:
		d.Inputs++
	case atom.H1:
		d.H1++
		d.Headings++
	case atom.H2, atom.H3, atom.H4, atom.H5, atom.H6:
		d.Headings++
	case atom.P:
		d.Paragraphs++
	case atom.Li:
		d.ListItems++
	case atom.Table:
		d.Tables++
	}
	if isHiddenStyle(attr(n, "style")) || attrPresent(n, "hidden") {
		d.HiddenStyles++
		inHidden = true
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walk(c, d, text, hidden, inHidden)
	}
}

func collectText(n *html.Node, sb *strings.Builder) {
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if c.Type == html.TextNode {
			sb.WriteString(c.Data)
			sb.WriteByte(' ')
		} else if c.Type == html.ElementNode {
			collectText(c, sb)
		}
	}
}

func isHiddenStyle(style string) bool {
	if style == "" {
		return false
	}
	s := strings.ToLower(strings.ReplaceAll(style, " ", ""))
	return strings.Contains(s, "display:none") ||
		strings.Contains(s, "visibility:hidden") ||
		strings.Contains(s, "font-size:0") ||
		strings.Contains(s, "opacity:0;") || strings.HasSuffix(s, "opacity:0") ||
		strings.Contains(s, "left:-") || strings.Contains(s, "text-indent:-")
}

func attr(n *html.Node, key string) string {
	for _, a := range n.Attr {
		if strings.EqualFold(a.Key, key) {
			return a.Val
		}
	}
	return ""
}

func attrPresent(n *html.Node, key string) bool {
	for _, a := range n.Attr {
		if strings.EqualFold(a.Key, key) {
			return true
		}
	}
	return false
}

func normalizeSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// Tokenize lowercases s and splits it into word tokens. Runs of digits are
// collapsed to "#num" so that phone numbers and prices do not explode the
// vocabulary. Tokens are capped at 30 runes to keep unsegmented CJK text
// from producing one enormous "word".
func Tokenize(s string) []string {
	var out []string
	var cur []rune
	flush := func() {
		if len(cur) == 0 {
			return
		}
		if len(cur) >= 2 || !unicode.IsLetter(cur[0]) {
			allDigit := true
			for _, r := range cur {
				if !unicode.IsDigit(r) {
					allDigit = false
					break
				}
			}
			if allDigit {
				out = append(out, "#num")
			} else if len(cur) <= 30 {
				out = append(out, string(cur))
			}
		}
		cur = cur[:0]
	}
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur = append(cur, r)
		} else if r == '\'' || r == '’' {
			continue
		} else {
			flush()
		}
	}
	flush()
	return out
}

// Host extracts the lowercased host from a URL-ish string, or "" if none.
func Host(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" || strings.HasPrefix(raw, "#") || strings.HasPrefix(raw, "javascript:") || strings.HasPrefix(raw, "mailto:") {
		return ""
	}
	if strings.HasPrefix(raw, "//") {
		raw = "http:" + raw
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return ""
	}
	h := strings.ToLower(u.Hostname())
	return strings.TrimPrefix(h, "www.")
}

// RegistrableDomain is a cheap approximation of eTLD+1: the last two labels,
// or three when the second-level label is a well-known generic one (co.uk).
func RegistrableDomain(host string) string {
	parts := strings.Split(host, ".")
	if len(parts) <= 2 {
		return host
	}
	second := parts[len(parts)-2]
	if len(parts) >= 3 && (second == "co" || second == "com" || second == "net" || second == "org" || second == "ac" || second == "gov") {
		return strings.Join(parts[len(parts)-3:], ".")
	}
	return strings.Join(parts[len(parts)-2:], ".")
}
