require(vtrackR)
require(GenomicRanges)

hush = suppressWarnings

load("../dmel_r5.57_FB2014_03/exons_r5.57.rda")
load("../dmel_r5.57_FB2014_03/act_exons_r5.57.rda")
load("../dmel_r5.57_FB2014_03/regreg_r5.57.rda")
load("../dmel_r5.57_FB2014_03/prom_r5.57.rda")
load("../dmel_r5.57_FB2014_03/genes_r5.57.rda")
load("../dmel_r5.57_FB2014_03/act_genes_r5.57.rda")
load("../dmel_r5.57_FB2014_03/introns_r5.57.rda")
load("../dmel_r5.57_FB2014_03/act_introns_r5.57.rda")

# Check that the files have not been tampered.
stopifnot(vcheck(exons_r5.57))
stopifnot(vcheck(act_exons_r5.57))
stopifnot(vcheck(regreg_r5.57))
stopifnot(vcheck(prom_r5.57))
stopifnot(vcheck(genes_r5.57))
stopifnot(vcheck(act_genes_r5.57))
stopifnot(vcheck(introns_r5.57))
stopifnot(vcheck(act_introns_r5.57))

p14 = subset(read.delim("../allprom.txt", comm="#",
   stringsAsFactors=FALSE), prom=="p14")
p14ab = list(p14, subset(p14, nexp > quantile(nexp, .95)))

#for (p14 in p14ab) {
for (p14 in p14ab[2]) {

   p14. = GRanges(Rle(p14$chr),
            IRanges(start=p14$pos, width=1), Rle(p14$strand))

#   for (ignore.strand in c(TRUE, FALSE)) {
   for (ignore.strand in FALSE) {

      N = nrow(p14)
      map14 = function(x) {
         findOverlaps(p14., x, ignore.strand=ignore.strand)
      }

      hush(ov_exons <- map14(exons_r5.57))
      hush(ov_act_exons <- map14(act_exons_r5.57))
      hush(ov_regreg <- map14(regreg_r5.57))
      hush(ov_genes <- map14(genes_r5.57))
      hush(ov_act_genes <- map14(act_genes_r5.57))
      hush(ov_prom <- map14(prom_r5.57))
      hush(ov_introns <- map14(introns_r5.57))
      hush(ov_act_introns <- map14(act_introns_r5.57))

#      print(nrow(as.matrix(ov_exons)) / N)
#      print(nrow(as.matrix(ov_act_exons)) / N)
#      print(nrow(as.matrix(ov_regreg)) / N)
#      print(nrow(as.matrix(ov_genes)) / N)
#      print(nrow(as.matrix(ov_act_genes)) / N)
#      print(nrow(as.matrix(ov_prom)) / N)
#      print(nrow(as.matrix(ov_introns)) / N)
#      print(nrow(as.matrix(ov_act_introns)) / N)

   }
}


# Count the number of exons for each gene.
#pairs = strsplit(sub('Name=([^;]*);.*', '\\1', act_exons_r5.57$attr), ':')
#exon_counts = tapply(X=as.integer(sapply(pairs, "[", 2)),
#   INDEX=sapply(pairs, "[", 1), max)
#hit_pairs = strsplit(sub('Name=([^;]*);.*', '\\1',
#   act_exons_r5.57[ov_act_exons@subjectHits]$attr), ':')
#position = as.integer(sapply(hit_pairs, "[", 2)) /
#   exon_counts[sapply(hit_pairs, "[", 1)]

# Get metagene representation.
strand = genes_r5.57[ov_genes@subjectHits]@strand
pranges = genes_r5.57[ov_genes@subjectHits][strand == '+']@ranges
mranges = genes_r5.57[ov_genes@subjectHits][strand == '-']@ranges
prelpos = p14.[ov_genes@queryHits][strand == '+']@ranges@start -
   pranges@start
mrelpos = mranges@start+mranges@width -
   p14.[ov_genes@queryHits][strand == '-']@ranges@start
fwdrelpos = c(prelpos / pranges@width, mrelpos / mranges@width)

# Invert the strands and map again.
p14$strand[p14$strand == '+'] = 'tmp'
p14$strand[p14$strand == '-'] = '+'
p14$strand[p14$strand == 'tmp'] = '-'
p14. = GRanges(Rle(p14$chr),
         IRanges(start=p14$pos, width=1), Rle(p14$strand))
hush(ov_genes <- findOverlaps(p14., genes_r5.57, ignore.strand=FALSE)

strand = genes_r5.57[ov_genes@subjectHits]@strand
pranges = genes_r5.57[ov_genes@subjectHits][strand == '+']@ranges
mranges = genes_r5.57[ov_genes@subjectHits][strand == '-']@ranges
prelpos = p14.[ov_genes@queryHits][strand == '+']@ranges@start -
   pranges@start
mrelpos = mranges@start+mranges@width -
   p14.[ov_genes@queryHits][strand == '-']@ranges@start
revrelpos = c(prelpos / pranges@width, mrelpos / mranges@width)

# Draw the figure.
L = .02
U = .0255555
length = .05
findy = function(x, drawnx, drawny) {
      keep = (drawnx > x-L) & (drawnx < x+L)
   drawny = drawny[keep]
      return (min(setdiff(seq(0, 2, by=U), drawny)))
}

pdf('~/Dropbox/arrows.pdf')
plot(c(0,1), c(-1,1.6), type='n', xaxt="n", yaxt="n", bty="n",
  xlab="", ylab="")
drawnx = c()
drawny = c()
for (x in fwdrelpos) {
   y = findy(x, drawnx, drawny)
   arrows(x-L, y, x, y, length=length, col=2)
   drawnx = c(drawnx, x)
   drawny = c(drawny, y)
}
drawnx = c()
drawny = c()
for (x in revrelpos) {
   y = findy(x, drawnx, drawny)
   arrows(x-L, -y, x, -y, length=length, col=4, code=1)
   drawnx = c(drawnx, x)
   drawny = c(drawny, y)
}
dev.off()
