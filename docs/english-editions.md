# English editions

The `blog` branch contains the published Jekyll source. The `main` branch only contains the repository README. Open changes against `blog`.

## Structure

- Chinese originals stay in `_posts/`, at their existing URLs. The main archive and RSS feed continue to contain originals only.
- English editions live in `_translations/`. For example, `_translations/tproxy.md` becomes `/en/tproxy/`.
- `/` is the Chinese homepage; `/en/` is the English homepage. `/posts` and `/en/posts/` list their respective editions.
- Homepage language selection is a normal link. The URL determines the language; browser language and local storage do not override shared links. Reading and language navigation work without JavaScript.
- Both article editions share a `translation_key`. The layouts resolve the corresponding article, render reciprocal language links, and add `hreflang` metadata. Each page retains its own canonical URL; the Chinese article remains the editorial source of truth.

## Coverage

All 12 published articles have English editions. The withdrawn data-structures entry remains in the Chinese archive with its existing withdrawal notice. The complete seven-article deep-learning series is indexed at `/en/series/the-tiny-learner/`.

The final batch and its technical clarifications are documented in [final-translations-review.md](final-translations-review.md).

## Add a translation

1. Add a unique, stable `translation_key` to the original's front matter, such as `tproxy`. Do not change its filename, date, categories, or permalink.
2. Create `_translations/<slug>.md` with English `title` and `description`, the same `translation_key`, the original `date`, and a `translated_at` date. The collection supplies `layout: default` and `lang: en`.
3. Translate the full article. Keep commands, paths, options, addresses, and mathematical notation accurate. Translate explanatory comments. Use `{% post_url original-filename-without-extension %}` for untranslated source articles and label those links as Chinese. Link to the English edition where one exists.
4. Check the translation against the original, including code blocks and the meaning of technical claims. Record any technical corrections separately for the author to review; do not silently revise the Chinese source.
5. Run `bundle exec jekyll build` and `bundle exec ruby scripts/check_editions.rb` before merging.

The English archive discovers new translations automatically. To keep an unfinished translation out of the generated site, set `published: false`; it will also be excluded from the archive and language switches. Keep `translation_key` unique within each collection, with exactly one original and one English edition per key. Do not copy `redirect_from` or the Chinese permalink into a translation.

## Initial translation review notes

The initial editions cover Linux TProxy and configuring Arch Linux as a router. Their article dates retain the originals' dates.

These are translations with a few explicit technical clarifications, not a new lab validation of the original configurations:

- **TProxy:** describe `SO_ORIGINAL_DST` in terms of connection tracking, rather than attributing destination recovery to the TCP handshake. Preserve the kernel documentation's race caveat. Avoid the original's absolute assertion that original UDP destination information is always irretrievably lost; the central distinction is that REDIRECT rewrites the packet while TProxy does not.
- **TProxy:** make clear that the shown policy-routing commands cover IPv4 even though the nftables table is `inet`. The local-output explanation is scoped to locally generated traffic; incoming traffic reaches PREROUTING before routing. Exempting the proxy's own traffic remains necessary.
- **Router:** move the MAC-address placeholder comment onto its own line so it is not part of the `PermanentMACAddress` value. Explain that `.link` naming is applied through udev, not a live networkd rename operation.
- **Router:** use the concrete IPv4 masquerading option for the IPv4 discussion. Keep the original network configuration blocks, but explain that working IPv6 also depends on upstream prefix delegation. Retain the original nftables example; it is not presented as a complete firewall policy.
- **Router:** replace the typographic SNAT placeholder with an explicitly non-executable address placeholder. Explain reply translation through connection tracking. Qualify the suggestion that a transparent proxy can replace masquerading: that depends on the traffic and protocols the proxy supports.

Review these differences and decide whether to make corresponding corrections to the Chinese originals separately.

Technical references checked while preparing these clarifications:

- https://docs.kernel.org/networking/tproxy.html
- https://github.com/systemd/systemd/blob/main/man/systemd.link.xml
- https://github.com/systemd/systemd/blob/main/man/systemd.network.xml

## Implementation checks

The regression check runs against generated HTML: original article URLs, separate archives, reciprocal edition links, language metadata, self-canonical URLs, and internal links on the English pages. GitHub Actions runs the same build and check for pull requests against `blog`; it does not deploy the site.
