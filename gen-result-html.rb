#!/usr/bin/env ruby

require 'ostruct'

load './SimpleHTML.rb'

$opts = OpenStruct.new

MissInfo = Struct.new(:outn, :except, :pred, :path)
LogInfo = Struct.new(:total, :miss_cnt, :outlist)

def read_filter_log(logfile)
  list = []
  miss_cnt = total = 0
  File.open(logfile, 'r:utf-8').each do |line|
    if line =~ /^Out(\d): expect (\S+), pred (.+), (\S+)$/
      list << MissInfo.new($1.to_i, $2, $3, $4)
    elsif line =~ /^Total (\d+)\/(\d+), /
      miss_cnt, total = $1.to_i, $2.to_i
    end
  end
  LogInfo.new(total, miss_cnt, list)
end

def gen_profile_table(html, loginfo)
  prof_arr = Array.new(6, 0)
  loginfo.outlist.each{ |r| prof_arr[r.outn] += 1 }
  total = loginfo.total
  html._table('border="1"') {
    put_table_head %w[Type Num Rate]
    prof_arr.each.with_index{ |n,i|
      _tr {
        _td("Out#{i}")
        _td(n.to_s)
        _td('%.2f%%' % (n.to_f * 100 / total))
      }
    }
  }
end

def gen_html(fname, loginfo)
  fout = File.open(fname, 'w:utf-8') or fail "open file failed"
  html = SimpleHTML.new fout
  html.head "HTML test"
  html.body {
    _p "HTML test page"
    _hr
    gen_profile_table(html, loginfo)
    _hr
    _table('border="1"') {
      put_table_head %w[OutN Expect Predict Picture]
      loginfo.outlist.each { |res|
        #next if res.outn < 3
        bgcolor = case res.outn
                  when 0..2 then 'green'
                  when 3..4 then 'orange'
                  else 'red'
                  end
        path = 'http://10.174.31.31/bruce/aiml/handwriting/' + res.path
        _tr {
          _td('Out' + res.outn.to_s, "bgcolor=\"#{bgcolor}\"")
          _td(res.except)
          _td(res.pred)
          _td{ _img(path) }
        }
      }
    }
  }
  html.close
end

info = read_filter_log(ARGV[0])
gen_html('test.html', info)

