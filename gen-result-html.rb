#!/usr/bin/env ruby

require 'ostruct'
require 'optparse'

$bin_dir = File.expand_path('..', __FILE__)

load $bin_dir + '/SimpleHTML.rb'


$opts = OpenStruct.new

MissInfo = Struct.new(:outn, :expect, :pred, :path)
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
  prof_arr[0] = total - loginfo.miss_cnt
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
  miss_tab = {}
  loginfo.outlist.each do |r|
    expt = r.expect
    miss_tab[expt] ||= { miss: 0, outn: [ 0, 0, 0, 0, 0, 0 ] }
    miss_tab[expt][:miss] += 1
    miss_tab[expt][:outn][r.outn] += 1
  end
  html._hr
  html._p "Miss Predicted word: #{miss_tab.size}"
  html._table('border="1"') {
    put_table_head %w[Word Miss Out1 Out2 Out3 Out4 Out5]
    miss_tab.each { |expt, val|
      next if val[:miss] < 3
      _tr {
        bgcolor = case val[:miss]
                  when 0..2 then 'green'
                  when 3..5 then 'orange'
                  else 'red'
                  end
        _td(expt, "bgcolor=\"#{bgcolor}\"")
        _td(val[:miss].to_s)
        val[:outn].last(5).each{|n| _td(n.to_s) }
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
        next if $opts.outn and res.outn < $opts.outn
        bgcolor = case res.outn
                  when 0..2 then 'green'
                  when 3..4 then 'orange'
                  else 'red'
                  end
        #path = 'http://10.174.31.31/bruce/aiml/handwriting/' + res.path
        path = $opts.img_root + '/' + res.path
        _tr {
          _td('Out' + res.outn.to_s, "bgcolor=\"#{bgcolor}\"")
          _td(res.expect)
          _td(res.pred)
          _td{ _img(path) }
        }
      }
    }
  }
  html.close
end

# main start
OptionParser.new do |opts|
  opts.banner = "gen-result-html.rb [options] result.log"
  opts.on('--img-root DIR', 'specify IMG ROOT in html') { |d| $opts.img_root = d }
  opts.on('--out HTML', 'specify output file') { |f| $opts.out = f }
  opts.on('--outn N', 'only print miss which out N') { |n| $opts.outn = n.to_i }
end.parse!

fail "need log file" if ARGV.empty?
logfile = ARGV[0]

$opts.out ||= '/dev/stdout'
$opts.img_root ||= Dir.pwd

info = read_filter_log(logfile)
gen_html($opts.out, info)

