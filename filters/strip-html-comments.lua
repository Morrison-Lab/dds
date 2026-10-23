-- HTML and revealjs: drop HTML comments from the rendered page.
--
-- Pandoc passes an HTML comment in a chapter through to the HTML output as
-- raw HTML, so author-facing notes, such as the pointer that marks where an
-- outtake was removed (see `chapters/outtakes/` in CLAUDE.md), are served in every
-- page's markup. A raw element is dropped only when its whole text
-- is comments and whitespace, so a raw HTML block that carries real markup
-- next to a comment is left alone.
--
-- Applied to the `html` and `revealjs` formats, in _quarto-website.yml.

local function only_comments(text)
  local rest = text:gsub('<!%-%-.-%-%->', '')
  return rest ~= text and rest:match('^%s*$') ~= nil
end

local function is_html(format)
  return format == 'html' or format == 'html5' or format == 'revealjs'
end

function RawBlock(el)
  if is_html(el.format) and only_comments(el.text) then
    return {}
  end
end

function RawInline(el)
  if is_html(el.format) and only_comments(el.text) then
    return {}
  end
end
