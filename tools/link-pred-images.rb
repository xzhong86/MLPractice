#!/usr/bin/env ruby

require 'fileutils'

fail "need image path" if ARGV.empty?

images_dir = ARGV[0]

images = []
Dir.glob(images_dir + '/*').each do |dir|
  images << Dir.glob(dir + '/*.png').sample
end
puts "get #{images.size} images"

images.first(500).shuffle.each.with_index do |img, idx|
  FileUtils.ln_s(img, idx.to_s + '.png')
end
