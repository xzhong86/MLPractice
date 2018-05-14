
# A simple HTML framework to generate HTML file
# BruceZ / zhongzhiping

class SimpleHTML
  def initialize(out)
    @out = out
    @out.puts '<!DOCTYPE html>'
    @out.puts '<html>'
    @depth = 1
  end
  def puts(str)
    @out.puts '  ' * @depth + str
  end
  def close
    @out.puts '</html>'
    @out.close
  end
  alias_method :_puts, :puts

  def tag_wrap(tag, _attr = nil, text = nil, &blk)
    attr_str = ''
    if _attr
      if _attr.kind_of? String
        attr_str = ' ' + _attr
      elsif _attr.kind_of? Hash
        attr_str = _attr.each.map{ |k,v| "#{k}=\"#{v}\"" }
      else
        fail
      end
    end
    _beg = '<' + tag + attr_str + '>'
    _end = '</' + tag + '>'
    wrap(_beg, _end, text, &blk)
    self
  end
  alias_method :_tag, :tag_wrap

  def wrap(_beg, _end, text = nil, &blk)
    text ||= ""
    if blk
      _puts _beg + text.to_s
      @depth += 1
      instance_eval &blk
      @depth -= 1
      _puts _end
    else
      _puts _beg + text.to_s + _end
    end
  end

  def head(str)
    tag_wrap('head') {
      _puts '<meta charset="utf-8">'
      _puts "<title>#{str}</title>"
    }
  end
  def body(attr = nil, &blk)
    tag_wrap('body', attr, &blk)
  end

  def put_ulist(lst)
    _ul { lst.each{ |e| _li e } }
  end
  def put_olist(lst)
    _ol { lst.each{ |e| _li w } }
  end
  def put_table(head, datas)
    _table {
      _tr { head.each{ |h| _th h } }
      datas.each{ |row|
        _tr { row.each{ |d| _td d } }
      }
    }
  end
  def put_table_head(head)
    _tr { head.each{ |h| _th h } }
  end


  def _img(url, att = '')
    _puts "<img src=\"#{url}\" #{att} />"
  end
  def _link(url, text = nil, &blk)
    tag_wrap('a', 'href=' + url, text, &blk)
  end
  def _hr
    _puts '<hr/>'
  end

  # attr as argument
  %w[table tr ul ol div span].each do |tag|
    define_method("_#{tag}") do |attr = nil, &blk|
      tag_wrap(tag, attr, nil, &blk)
    end
  end
  # text as first argument
  %w[p th td li].each do |tag|
    define_method("_#{tag}") do |text = nil, att = nil, &blk|
      tag_wrap(tag, att, text, &blk)
    end
  end

end

# _table {
#   _tr { head.each{ |h| _th h } }
#   _tr { col0.each{ |d| _td d } }
#   cols.each { |col|
#     _tr { col.each{ |d| _td d } }
#   }
# }
