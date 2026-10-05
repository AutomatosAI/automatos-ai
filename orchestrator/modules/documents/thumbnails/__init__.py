"""F353 (issue #947): a document Deliverable's card shows its first page.

Images on the Deliverables page showed the picture; PDFs, Word documents, reports
and spreadsheets showed a stock icon. Each such Deliverable now gets a small
picture of its first page, drawn when it is registered and stored beside the
generated documents:

* ``eligibility``  which Deliverables get one, and the URL their card asks;
* ``render``       draws page 1 as a PNG (pypdfium2), in a child process;
* ``html_sources`` turns a Word document, a sheet, a CSV or markdown into a page;
* ``sources``      reads the document's bytes where it lives;
* ``store``        keeps the PNG next to the generated documents (disk + S3);
* ``job``          renders one Deliverable and records the result on its row;
* ``schedule``     queues that job off the request when a Deliverable is made;
* ``backfill``     renders the ones made before this, at a set rate.

A failed render leaves the card's icon in place and logs the reason; it never
fails or delays the document itself.
"""
