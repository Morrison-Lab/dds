-- PDF handouts: turn links to other pages of the site into plain text,
-- and links to files the site publishes into links to the site.
--
-- Each PDF is shared on its own (on Canvas), so a link to another chapter,
-- or to an anchor on another chapter, has nowhere to go. Keep the link text
-- and drop the link. An empty target goes nowhere either, so it is dropped
-- too. Links with a scheme (https:, mailto:), protocol-relative links
-- (//host/path) and anchors on the same page (#sec-...) are left alone.
--
-- A relative link to a file that is not a page, such as `data/auto.csv`,
-- points at a copy the site publishes through the page's `resources:`.
-- Students can reach that copy on the site but not in the repository,
-- so the link is rewritten to its absolute URL under `handout-site-url`,
-- which _quarto-website.yml sets to the website's `site-url`.
--
-- Applied only to the `pdf` format, in _quarto-website.yml.

local site_url = nil

local function is_relative(target)
  if target:sub(1, 1) == '#' or target:sub(1, 2) == '//' then
    return false
  end
  -- A URI scheme is letters, digits, "+", "-" or "." before a colon.
  if target:match('^%a[%w+.-]*:') then
    return false
  end
  return true
end

-- A target naming a page of the site, or nothing at all.
local function is_page(target)
  local path = target:gsub('[#?].*$', '')
  return path == '' or path:match('/$') ~= nil
    or path:match('%.qmd$') ~= nil
    or path:match('%.html?$') ~= nil
    or path:match('%.md$') ~= nil
end

-- Join a directory and a relative path, resolving "." and "..".
local function join(dir, target)
  local parts = {}
  for part in (dir .. '/' .. target):gmatch('[^/]+') do
    if part == '..' then
      table.remove(parts)
    elseif part ~= '.' then
      table.insert(parts, part)
    end
  end
  return table.concat(parts, '/')
end

-- The directory of the page being rendered, relative to the project root.
local function page_dir()
  local input = quarto.doc.input_file
  local root = quarto.project.directory
  if not input or not root then
    return nil
  end
  if input:sub(1, #root + 1) ~= root .. '/' then
    return nil
  end
  return pandoc.path.directory(input:sub(#root + 2))
end

local function read_site_url(meta)
  local url = meta['handout-site-url']
  if url then
    url = pandoc.utils.stringify(url)
    if url:sub(-1) ~= '/' then
      url = url .. '/'
    end
    site_url = url
  end
end

local function Link(el)
  if not is_relative(el.target) then
    return nil
  end
  if not is_page(el.target) and site_url then
    local dir = page_dir()
    if dir then
      el.target = site_url .. join(dir, el.target)
      return el
    end
  end
  return el.content
end

return {
  { Meta = read_site_url },
  { Link = Link },
}
