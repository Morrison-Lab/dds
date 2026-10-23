-- Add a "Webpage version" link to every reveal.js deck, pointing at the
-- matching section of the deck's own HTML page.
--
-- The deck and the page are two renders of one `.qmd`, so a slide's section
-- id is also the page's section id -- with one exception the id-lookup below
-- has to exclude: Quarto names a deck's opening section `title-slide`, while
-- the HTML page names its own `title-block-header`, so `#title-slide` would
-- resolve to nothing.

local SCRIPT = [[
<script>
(function () {
  var css = [
    "#revealjs-html-link {",
    "  position: fixed;",
    "  top: 8px;",
    "  right: 8px;",
    "  z-index: 31;",
    // The element is a child of <body>, outside `.reveal`, so an `em` here
    // resolves against the 16px UA default rather than the deck's 40px
    // `--r-main-font-size`. An absolute size sidesteps that question
    // entirely, which is why this is px rather than em.
    "  font-size: 12px;",
    "  line-height: 1;",
    "}",
    "#revealjs-html-link a {",
    // The pill composites over the deck theme's #191919 background, so the
    // resting text colour decides the contrast a reader actually sits in.
    // At 0.5 alpha it composites to #686868 on #D1D1D1 -- 3.65:1, under the
    // 4.5:1 WCAG 2.1 AA floor for 12px text. 0.75 clears it.
    "  color: rgba(0, 0, 0, 0.75);",
    "  text-decoration: none;",
    "  border: 1px solid rgba(0, 0, 0, 0.2);",
    "  padding: 4px 8px;",
    "  border-radius: 4px;",
    "  background: rgba(255, 255, 255, 0.8);",
    "  transition: color 0.15s, background 0.15s;",
    "}",
    "#revealjs-html-link a:hover {",
    "  color: rgba(0, 0, 0, 0.9);",
    "  background: rgba(255, 255, 255, 0.98);",
    "  border-color: rgba(0, 0, 0, 0.4);",
    "}",
    // The element is a child of <body>, outside `.reveal`. Two selectors in
    // reveal.css's own @media print block do reach outside `.reveal`
    // (`html:not(.print-pdf)` and `html:not(.print-pdf) body`), but both set
    // only overflow/size/margin, so nothing there hides this element. Without
    // these two rules a ?print-pdf export renders the badge on every page,
    // each copy carrying whichever fragment was current when the export
    // began.
    "@media print {",
    "  #revealjs-html-link { display: none; }",
    "}",
    ".print-pdf #revealjs-html-link {",
    "  display: none;",
    "}"
  ].join("\n");

  // Quarto gives every deck's opening section this id; no HTML page has it.
  var TITLE_SLIDE_ID = "title-slide";

  // The current URL with the fragment and any query string removed. Both have
  // to come off before the suffix test below, because that test is anchored at
  // the end of the string -- with `?foo` still attached, a deck following the
  // convention reads as one that does not.
  function currentPath() {
    return window.location.href.split("#")[0].split("?")[0];
  }

  // The deck's HTML-page URL, or null when this deck's filename does not
  // follow the `-slides.html` convention the mapping depends on.
  //
  // The query is dropped rather than carried over: every query a Quarto deck
  // carries is a reveal view-mode switch (`?print-pdf`, `?view=scroll`) that
  // means nothing on the HTML page.
  function getHtmlBase() {
    var path = currentPath();
    var htmlPath = path.replace(/-slides\.html$/, ".html");
    if (htmlPath === path) {
      return null;
    }
    return htmlPath;
  }

  function getCurrentSectionId() {
    if (typeof Reveal === "undefined" || !Reveal.getCurrentSlide) {
      return null;
    }
    var slide = Reveal.getCurrentSlide();
    if (!slide) {
      return null;
    }
    // If the current slide itself has a usable id, use it.
    if (slide.id && slide.id !== TITLE_SLIDE_ID) {
      return slide.id;
    }
    // Otherwise walk BACKWARDS through sibling sections to the nearest one
    // with a usable id -- backwards, so the link never points past content
    // the viewer has not reached.
    var sibling = slide.previousElementSibling;
    while (sibling) {
      if (sibling.tagName === "SECTION" && sibling.id && sibling.id !== TITLE_SLIDE_ID) {
        return sibling.id;
      }
      sibling = sibling.previousElementSibling;
    }
    // No section at or before this slide has an id, so link to the page
    // itself rather than to a section the viewer has not reached.
    return null;
  }

  function updateHtmlLink() {
    // Site 2 of the four holding this identifier; see the comment on
    // `anchor.id` in initRevealJsHtmlLink below before renaming it. A rename
    // that misses this line leaves the lookup failing on every slide change,
    // so the link never targets a section and nothing reports it.
    var anchor = document.getElementById("revealjs-html-anchor");
    if (!anchor) {
      return;
    }
    var htmlBase = getHtmlBase();
    if (!htmlBase) {
      return;
    }
    var sectionId = getCurrentSectionId();
    anchor.href = htmlBase + (sectionId ? "#" + sectionId : "");
  }

  function initRevealJsHtmlLink() {
    var htmlBase = getHtmlBase();
    if (!htmlBase) {
      // Say so rather than vanishing: an absent link and a broken one are
      // indistinguishable from the page otherwise.
      console.warn(
        "revealjs-html-links: this deck's URL does not end in -slides.html, " +
        "so no webpage-version link was added."
      );
      return;
    }

    // Inject styles
    var styleElement = document.createElement("style");
    styleElement.textContent = css;
    document.head.appendChild(styleElement);

    // Create the link element and append it directly to body so it stays
    // visible across all slides regardless of Reveal.js section visibility
    var div = document.createElement("div");
    div.id = "revealjs-html-link";
    var anchor = document.createElement("a");
    // This identifier is written here and read at three other sites, and all
    // four have to be renamed together:
    //   1. this assignment, which puts the id on the element;
    //   2. `document.getElementById` in updateHtmlLink above, which finds it
    //      again on every slide change;
    //   3. `FILTER_MARKER` in `tools/check_site.py`, which greps it out of
    //      the rendered page to confirm the filter ran;
    //   4. the comment above that constant, which spells the identifier out
    //      as an `id=` attribute while explaining why the rendered page holds
    //      no such attribute.
    // Renaming every site but 2 is the silent failure: updateHtmlLink returns
    // early forever, so the link stops targeting the current section, while
    // this assignment still leaves the marker in the injected script --- once,
    // which is all check_site.py's membership test needs to stay green.
    anchor.id = "revealjs-html-anchor";
    anchor.href = htmlBase;
    anchor.target = "_blank";
    anchor.rel = "noopener";
    // The visible text is the accessible name, so a voice-control user can
    // say what they can see (WCAG 2.5.3, Label in Name). The fuller
    // description rides on `title` instead of overriding the name via
    // `aria-label`, which would break that match.
    // The new-tab warning rides on `title`, which reaches a mouse user on
    // hover and a screen-reader user whose reader surfaces the accessible
    // description on focus, and reaches a touch-only user not at all. So the
    // warning is present for some readers and absent for others. Whether that
    // satisfies the relevant WCAG technique, or whether the warning belongs
    // in the link text itself, is unsettled here rather than answered either
    // way: see Morrison-Lab/lds#73.
    // The document emoji is decoration, so it is hidden from the accessible
    // name rather than announced ahead of the words a speaker would say.
    anchor.title = "View this section in the HTML notes (opens in a new tab)";
    anchor.innerHTML =
      '<span aria-hidden="true">&#128196;&nbsp;</span>Webpage version';
    div.appendChild(anchor);
    document.body.appendChild(div);

    // Register Reveal.js slide-change listener to keep the link updated
    if (typeof Reveal !== "undefined") {
      Reveal.on("ready", updateHtmlLink);
      Reveal.on("slidechanged", updateHtmlLink);
    }

    updateHtmlLink();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initRevealJsHtmlLink);
  } else {
    initRevealJsHtmlLink();
  }
})();
</script>
]]

function Pandoc(doc)
  if not quarto.doc.is_format("revealjs") then
    return doc
  end
  -- `after-body` puts the script outside the slide tree. Appending a RawBlock
  -- to `doc.blocks` instead lands it inside the LAST slide's <section>, which
  -- Reveal loads, unloads, and marks aria-hidden.
  quarto.doc.include_text("after-body", SCRIPT)
  return doc
end
