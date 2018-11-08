use utf8;
if (@ARGV != 1) {
  print "usage: script wav.scp; output wav_cvt.scp to stdout.\n"
}
$wavscp = $ARGV[0];

sub trim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

# stereo to mono channel
# 44.1k to 16k
# mp3 to wav
if (open(my $fh_wavscp, "<", $wavscp)) {
  while (my $line = <$fh_wavscp>) {
    chomp $line;
    $line = trim($line);
    my($wavid,$wavpath) = split(" ", $line, 2);
    if ($wavpath =~ /\|$/) {
      # a command
      # assume input is already converted into wav file
      $wavpath .= " | sox -t wav - -r 16k -t wav - remix 1,2 |"
    } else {
      # a normal filepath
      $wavpath = join " ", "sox", $wavpath, "-r 16k -t wav - remix 1,2 |"
    }
    print "$wavid $wavpath\n"
  }
}
