---
title: 系列合集
permalink: /series/
series: true
---


{% comment %}
  Find all index pages under the series/ directory, excluding this page.
{% endcomment %}
{% assign series_pages = site.pages | where_exp: 'p', 'p.path contains "series/"' %}
{% assign series_indexes = series_pages | where: 'name', 'index.md' %}
{% assign series_indexes = series_indexes | where_exp: 'p', 'p.url != page.url' %}
{% assign series_list = series_indexes | sort: 'title' %}

{% if series_list and series_list.size > 0 %}
<ul class="post-list" style="list-style: none; padding-left: 0;">
  {% for sp in series_list %}
  <li style="margin-bottom: 1.2em;">
    <h2 style="margin: 0;">
      <a class="post-link" href="{{ sp.url | relative_url }}">{{ sp.title | default: sp.url }}</a>
    </h2>
    {% if sp.subtitle %}
      <span class="post-meta">{{ sp.subtitle }}</span>
    {% endif %}
  </li>
  {% endfor %}
</ul>
{% else %}
<p>暂未发现系列入口页。请在 <code>series/</code> 目录下创建各系列的 <code>index.md</code>。</p>
{% endif %}
