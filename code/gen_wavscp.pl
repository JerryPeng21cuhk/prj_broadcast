use utf8;
if (@ARGV != 1) {
  print "usage: script wavlist; output wav.scp to stdout.\n"
}
$wavlist = $ARGV[0];

sub trim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  $string =~ s/\s+$//;
  return $string;
}

#only select those audio files matching to predefined format.
# and use them to create wav.scp

# 傍晚 -- \x{508D}\x{665A}
# 午間 -- \x{5348}\x{9593}
# 晨早 -- \x{6668}\x{65E9}
# 晚間 -- \x{665A}\x{9593}

if (open(my $fh_wavlist, "<", $wavlist)) {
  while (my $line = <$fh_wavlist>) {
    chomp $line;
    $line = trim($line);
    if ( $line =~ /(\d{8})_?\x{508D}\x{665A}/ ) {
      print "$1-evening $line\n";
    }
    if ( $line =~ /(\d{8})_?\x{5348}\x{9593}/ ) {
      print "$1-noon $line\n";
    }
    if ( $line =~ /(\d{8})_?\x{6668}\x{65E9}/ ) {
      print "$1-morning $line\n";
    }
    if ( $line =~ /(\d{8})_?\x{665A}\x{9593}/ ) {
      print "$1-night $line\n";
    }
    
  }
}
