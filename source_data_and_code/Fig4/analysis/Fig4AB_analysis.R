# Fig. 4A, B: GO and Disease Ontology enrichment of genes carrying a gnomAD v4.0 SNV with
# |delta pTE| > 0.1 in the 5'UTR, 3'UTR or CDS (synonymous or missense).
#   A  GO terms (all ontologies) reduced to parent terms by semantic similarity (rrvgo, BP, threshold 0.9)
#   B  Disease Ontology terms and their similarity network
suppressPackageStartupMessages({
  library(data.table)
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(DOSE)
  library(enrichplot)
  library(rrvgo)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
base_dir <- if (length(file_arg)) dirname(dirname(normalizePath(sub("^--file=", "", file_arg)))) else getwd()
input_dir <- file.path(base_dir, "input")
output_dir <- file.path(base_dir, "output")
dir.create(output_dir, showWarnings = FALSE)

# SNV-transcript pairs with |delta pTE| > 0.1 in the four regions, with the gene of each transcript
# (built by analysis/Fig4AB_input_subset.R)
pairs <- fread(file.path(input_dir, "Pred_All.gnomad.exomes.v4.0.sites.PASS_pc_SNV_NM.filter_cor0.7_l0.25_h1.TPE_opt.delta.abs_gt_0.1.genes.tsv"),
               sep = "\t", select = c("gene", "delta_score_high"))
sel_gene_filter <- unique(pairs$gene[abs(pairs$delta_score_high) > 0.1])
cat(sprintf("%d genes with |delta pTE| > 0.1\n", length(sel_gene_filter)))

## ---- Fig. 4A ----
go_var <- enrichGO(sel_gene_filter, OrgDb = org.Hs.eg.db, ont = "ALL", pAdjustMethod = "fdr",
                   pvalueCutoff = 0.05, qvalueCutoff = 0.05, keyType = "SYMBOL")
simMatrix <- calculateSimMatrix(go_var$ID, orgdb = "org.Hs.eg.db", ont = "BP", method = "Rel")
scores <- setNames(-log10(go_var$qvalue), go_var$ID)
reducedTerms <- reduceSimMatrix(simMatrix, scores, threshold = 0.9, orgdb = "org.Hs.eg.db")
cat(sprintf("GO: %d enriched terms, %d BP terms in %d parent terms\n",
            nrow(as.data.frame(go_var)), nrow(reducedTerms), length(unique(reducedTerms$parent))))

# 2-D embedding of the term similarity, as drawn by rrvgo::scatterPlot
x <- cmdscale(as.matrix(as.dist(1 - simMatrix)), eig = TRUE, k = 2)
coords <- cbind(as.data.frame(x$points), reducedTerms[match(rownames(x$points), reducedTerms$go), c("term", "parent", "parentTerm", "size")])

fwrite(as.data.frame(go_var), file.path(output_dir, "Fig4A_GO_enrichment.tsv"), sep = "\t")
fwrite(cbind(go = rownames(coords), coords), file.path(output_dir, "Fig4A_source_data.tsv"), sep = "\t")
saveRDS(list(simMatrix = simMatrix, reducedTerms = reducedTerms), file.path(output_dir, "Fig4A_rrvgo.rds"))

## ---- Fig. 4B ----
sel_gene_filter_ENTREZID <- unique(bitr(sel_gene_filter, fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db)$ENTREZID)
# DOSE >= 3.99 names the human Disease Ontology "HDO"
do_ont <- if (packageVersion("DOSE") >= "3.99") "HDO" else "DO"
DO_var <- enrichDO(gene = sel_gene_filter_ENTREZID, ont = do_ont, pvalueCutoff = 0.05, pAdjustMethod = "fdr",
                   qvalueCutoff = 0.05, readable = TRUE)
DO_var <- pairwise_termsim(DO_var)
cat(sprintf("DO: %d enriched terms\n", nrow(as.data.frame(DO_var))))

fwrite(as.data.frame(DO_var), file.path(output_dir, "Fig4B_source_data.tsv"), sep = "\t")
saveRDS(DO_var, file.path(output_dir, "Fig4B_DO_enrichment.rds"))
