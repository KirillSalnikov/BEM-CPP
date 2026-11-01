-- Break long inline filenames at punctuation without changing Markdown.
function Code(element)
  if FORMAT:match('latex') and not element.text:find('[{}\\]') then
    return pandoc.RawInline('latex', '\\nolinkurl{' .. element.text .. '}')
  end
end
