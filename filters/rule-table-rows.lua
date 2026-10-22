-- PDF handouts: draw a rule between the body rows of a table.
--
-- Pandoc's LaTeX tables have rules only above and below the header and at
-- the bottom, so a table whose rows each span several lines (the calendar's
-- weeks) runs together on paper. Wrap such a table in a `.ruled-rows` div
-- and this filter adds a booktabs \midrule after every body row but the last,
-- and a little space between the separate items within a cell.
--
-- The rules are inserted into the table's own LaTeX, so the filter renders
-- the table here and returns it as raw LaTeX. If the rendered table does not
-- end its body rows where expected, the table is left as pandoc wrote it and
-- a warning is printed, rather than guessing. Citations, images and spans in
-- the table are kept out of that raw LaTeX, so they are still processed.
--
-- Applied only to the `pdf` format, in _quarto-website.yml, and listed
-- after unlink-pages.lua: the table is written to LaTeX here, so its links
-- must already be plain text by the time this filter sees it.

local ROW_END = ' \\\\\n'
local COLUMN = '\\raggedright\\arraybackslash}p{'
local COLUMN_SPACED = '\\raggedright\\arraybackslash\\setlength{\\parskip}{3pt}}p{'

local function body_row_count(tbl)
  local n = 0
  for _, body in ipairs(tbl.bodies) do
    n = n + #body.body
  end
  return n
end

-- Citations, images and spans inside the table are not finished when this
-- filter runs: citeproc resolves citations after every filter, and Quarto
-- still has to process images and its own span-based nodes (such as a
-- shortcode's output). Written to LaTeX here, they would come out as their
-- source text (`[@key]`) or as nothing. So each is swapped for a placeholder
-- token before the table is written, and put back afterwards as itself,
-- between raw LaTeX pieces, where the later stages still see it.
local PLACEHOLDER = 'RuledRowsPlaceholder%dX'

local PLACEHOLDER_PATTERN = 'RuledRowsPlaceholder(%d+)X'

local function hold_inlines(tbl)
  local held = {}
  -- Top-down, and not into what is held: a citation inside a held span
  -- goes back with its span.
  local function hold(el)
    held[#held + 1] = el
    return pandoc.Str(PLACEHOLDER:format(#held)), false
  end
  tbl = tbl:walk({ traverse = 'topdown', Cite = hold, Image = hold, Span = hold })
  return tbl, held
end

local function restore_inlines(latex, held)
  if #held == 0 then
    return pandoc.RawBlock('latex', latex)
  end
  local inlines = {}
  local seen = 0
  local start = 1
  for a, i, b in latex:gmatch('()' .. PLACEHOLDER_PATTERN .. '()') do
    inlines[#inlines + 1] = pandoc.RawInline('latex', latex:sub(start, a - 1))
    inlines[#inlines + 1] = held[tonumber(i)]
    seen = seen + 1
    start = b
  end
  if seen ~= #held then
    return nil, 'found ' .. seen .. ' of ' .. #held
      .. ' placeholders in the table\'s LaTeX'
  end
  inlines[#inlines + 1] = pandoc.RawInline('latex', latex:sub(start))
  return pandoc.Plain(inlines)
end

local function rule_rows(tbl)
  local latex = pandoc.write(pandoc.Pandoc({ tbl }), 'latex')
  local head, body, tail = latex:match('^(.-\\endlastfoot\n)(.-)(\\end{longtable}.*)$')
  if not body then
    return nil, 'no longtable body found'
  end

  local rows = {}
  local start = 1
  while true do
    local i, j = body:find(ROW_END, start, true)
    if not i then
      break
    end
    rows[#rows + 1] = body:sub(start, j)
    start = j + 1
  end
  if start <= #body or #rows ~= body_row_count(tbl) then
    return nil, 'found ' .. #rows .. ' row endings for '
      .. body_row_count(tbl) .. ' body rows'
  end

  -- Items within a cell are separate paragraphs; space them apart so a
  -- wrapped item does not read as two. A \multicolumn cell carries its own
  -- column type inside its row, so the body needs the same substitution.
  local spaced, n = head:gsub(COLUMN, COLUMN_SPACED)
  if n == 0 then
    return nil, 'no raggedright column types found'
  end
  local ruled = table.concat(rows, '\\midrule\\noalign{}\n')
  ruled = ruled:gsub(COLUMN, COLUMN_SPACED)

  return spaced .. ruled .. tail
end

function Div(el)
  if not FORMAT:match('latex') or not el.classes:includes('ruled-rows') then
    return nil
  end
  local blocks = {}
  for _, block in ipairs(el.content) do
    if block.t == 'Table' then
      local tbl, held = hold_inlines(block)
      local latex, err = rule_rows(tbl)
      local ruled
      if latex then
        ruled, err = restore_inlines(latex, held)
      end
      if ruled then
        block = ruled
      else
        io.stderr:write('rule-table-rows.lua: ' .. err
          .. '; leaving the table unruled\n')
      end
    end
    blocks[#blocks + 1] = block
  end
  return blocks
end
