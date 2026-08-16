## Test environments

* local Windows 11, R 4.5.2
* R CMD check --as-cran

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Merwan Roudane <merwanroudane920@gmail.com>'
  New submission

This is the expected note on a first submission.

## Pre-flight

* References in DESCRIPTION are auto-linked with <doi:...>, no space after
  'doi:', in angle brackets. Every DOI was verified to resolve via doi.org and
  its bibliographic record checked against Crossref:
    - Villena (2026)              <doi:10.2139/ssrn.6472826>
    - Pesaran, Shin, Smith (2001) <doi:10.1002/jae.616>
    - Chernozhukov et al. (2018)  <doi:10.1111/ectj.12097>
    - Zou (2006)                  <doi:10.1198/016214506000000735>
    - McCracken and Ng (2016)     <doi:10.1080/07350015.2015.1086655>
* Acronyms are spelled out on first use: Autoregressive Distributed Lag (ARDL),
  Double Machine Learning (DML).
* Software and database names are single-quoted in the Description: 'FRED-MD'.
* Every exported function has at least one runnable example outside \donttest{}
  finishing in well under 5 seconds. Longer, realistic demonstrations (the
  bootstrap at useful B, the four-fit diagnostic) are inside \donttest{}.
* No \dontrun{} is used anywhere.
* Math in the Rd files is wrapped in \eqn{} and \deqn{}.
* inst/CITATION uses bibentry() with textVersion, and cites both the method
  paper and the software.
* The bundled dataset is public macroeconomic data (FRED-MD) and is documented
  with its source.
* The test suite checks each exported estimator, including exact-agreement
  regression tests against an independent implementation of the same procedure.

## Notes for the reviewer

The statistical procedure implemented here is due to Villena (2026); this
package is an independent implementation of it and cites the paper in the
Description, the package documentation, inst/CITATION and the vignette.
