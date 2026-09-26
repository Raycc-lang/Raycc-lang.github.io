# frozen_string_literal: true

require 'nokogiri'
require 'uri'
require 'yaml'
require 'date'
require 'time'

def check(condition, message)
  abort message unless condition
end

def front_matter(path)
  YAML.safe_load(File.read(path).split(/^---\s*$\n?/, 3)[1], permitted_classes: [Date, Time])
end

def output_file(url)
  path = URI.parse(url).path
  candidates = ["_site#{path}", "_site#{path}/index.html", "_site#{path}.html"]
  candidates.find { |candidate| File.file?(candidate) }
end

def html(url)
  file = output_file(url)
  check(file, "Missing generated page: #{url}")
  Nokogiri::HTML(File.read(file))
end

originals = Dir['_posts/*.md'].map { |path| [path, front_matter(path)] }
translations = Dir['_translations/*.md'].map { |path| [path, front_matter(path)] }
check(translations.map { |_, data| data['translation_key'] }.uniq.size == translations.size,
      'Duplicate English translation_key')

translations.each do |path, data|
  key = data.fetch('translation_key')
  matches = originals.select { |_, original| original['translation_key'] == key }
  check(matches.size == 1, "Expected exactly one Chinese original for #{key}")
  next if data['published'] == false

  url = "/en/#{File.basename(path, '.md')}/"
  english = html(url)
  check(english.at('html')['lang'] == 'en', "Incorrect lang: #{url}")
  notice = english.at('.translation-notice')&.text.to_s
  check(notice.include?('translated using AI'), "Missing AI disclosure: #{url}")
  expected = data['author_reviewed'] == true ? 'author has reviewed' : 'not yet been reviewed by the author'
  check(notice.include?(expected), "Incorrect author-review status: #{url}")
  check(URI.parse(english.at('link[rel="canonical"]')['href']).path == url, "Canonical changed: #{url}")

  original_url = english.at('link[hreflang="zh-CN"]')&.[]('href')
  check(original_url, "Missing Chinese alternate: #{url}")
  chinese = html(original_url)
  check(chinese.at('html')['lang'] == 'zh-CN', "Incorrect Chinese lang: #{original_url}")
  check(chinese.at('.translation-notice').nil?, "Disclosure incorrectly applied to original: #{original_url}")
  check(URI.parse(chinese.at('link[rel="canonical"]')['href']).path == URI.parse(original_url).path,
        "Chinese canonical changed: #{original_url}")
  check(URI.parse(chinese.at('link[hreflang="en"]')['href']).path == url, "Nonreciprocal alternate: #{url}")
  check(chinese.css('.edition-switch a').any? { |a| URI.parse(a['href']).path == url }, "Missing English switch: #{url}")
  check(english.css('.edition-switch a').any? { |a| URI.parse(a['href']).path == URI.parse(original_url).path }, "Missing Chinese switch: #{url}")
end

archive = html('/posts').css('a.post-link').map { |a| a['href'] }
check(archive.size == originals.size, 'Chinese archive lost posts or includes duplicates')
check(archive.none? { |url| url.start_with?('/en/') }, 'English translations leaked into Chinese archive')
archive.each { |url| check(output_file(url), "Missing original URL: #{url}") }
english_archive = html('/en/posts/').css('a.post-link').map { |a| a['href'] }
check(english_archive.size == translations.count { |_, data| data['published'] != false }, 'English archive count mismatch')
check(english_archive.all? { |url| url.start_with?('/en/') }, 'Chinese posts leaked into English archive')

%w[/ /en/ /en/posts/].each do |url|
  doc = html(url)
  check(doc.at('html')['lang'] == (url.start_with?('/en/') ? 'en' : 'zh-CN'), "Incorrect page language: #{url}")
end

Dir['_site/en/**/*.html'].each do |file|
  doc = Nokogiri::HTML(File.read(file))
  doc.css('a[href]').each do |link|
    uri = URI.parse(link['href'])
    next unless uri.path&.start_with?('/') && (uri.host.nil? || uri.host == 'raycc.org')
    target = output_file(uri.path)
    check(target, "Broken internal link in #{file}: #{link['href']}")
    if uri.fragment && target.end_with?('.html')
      target_doc = Nokogiri::HTML(File.read(target))
      check(target_doc.xpath('//*[@id]').any? { |node| node['id'] == uri.fragment }, "Missing anchor: #{link['href']}")
    end
  end
end

puts "English editions checked: #{english_archive.size} translations, #{archive.size} original posts; links, metadata, and disclosures pass."
