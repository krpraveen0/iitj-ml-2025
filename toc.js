// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><a href="introduction.html"><strong aria-hidden="true">1.</strong> Introduction</a></li><li class="chapter-item expanded "><a href="syllabus.html"><strong aria-hidden="true">2.</strong> Syllabus</a></li><li class="chapter-item expanded "><a href="evaluation.html"><strong aria-hidden="true">3.</strong> Evaluation Scheme</a></li><li class="chapter-item expanded "><a href="schedule.html"><strong aria-hidden="true">4.</strong> Schedule</a></li><li class="chapter-item expanded "><a href="resources.html"><strong aria-hidden="true">5.</strong> Resources</a></li><li class="chapter-item expanded "><a href="fractals/fractal1_overview.html"><strong aria-hidden="true">6.</strong> Fractal I: Supervised Learning</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="notes/lecture0.html"><strong aria-hidden="true">6.1.</strong> Lecture 0: Pre-Requisites for ML</a></li><li class="chapter-item expanded "><a href="notes/lecture1.html"><strong aria-hidden="true">6.2.</strong> Lecture 1: Introduction to AI and ML</a></li><li class="chapter-item expanded "><a href="notes/lecture2.html"><strong aria-hidden="true">6.3.</strong> Lecture 2: Paradigms of ML</a></li><li class="chapter-item expanded "><a href="notes/lecture3.html"><strong aria-hidden="true">6.4.</strong> Lecture 3: Bayesian &amp; Decision Trees</a></li><li class="chapter-item expanded "><a href="notes/lecture4.html"><strong aria-hidden="true">6.5.</strong> Lecture 4: Ensemble Methods</a></li></ol></li><li class="chapter-item expanded "><a href="fractals/fractal2_overview.html"><strong aria-hidden="true">7.</strong> Fractal II: Graphical Models &amp; Neural Networks</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="notes/lecture5.html"><strong aria-hidden="true">7.1.</strong> Lecture 5: Graphical Models (HMM, CRF)</a></li><li class="chapter-item expanded "><a href="notes/lecture6.html"><strong aria-hidden="true">7.2.</strong> Lecture 6: Perceptron &amp; Backprop</a></li><li class="chapter-item expanded "><a href="notes/lecture7.html"><strong aria-hidden="true">7.3.</strong> Lecture 7: RNN, LSTM, GRU</a></li><li class="chapter-item expanded "><a href="notes/lecture8.html"><strong aria-hidden="true">7.4.</strong> Lecture 8: Encoder-Decoder, Attention, GANs</a></li></ol></li><li class="chapter-item expanded "><a href="fractals/fractal3_overview.html"><strong aria-hidden="true">8.</strong> Fractal III: Unsupervised Learning</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="notes/lecture9.html"><strong aria-hidden="true">8.1.</strong> Lecture 9: Feature Selection &amp; PCA</a></li><li class="chapter-item expanded "><a href="notes/lecture10.html"><strong aria-hidden="true">8.2.</strong> Lecture 10: Clustering Methods</a></li><li class="chapter-item expanded "><a href="notes/lecture11.html"><strong aria-hidden="true">8.3.</strong> Lecture 11: Hypothesis Evaluation</a></li></ol></li><li class="chapter-item expanded "><a href="assignments/assignments.html"><strong aria-hidden="true">9.</strong> Assignments</a></li><li class="chapter-item expanded "><a href="notes/in_progress.html"><strong aria-hidden="true">10.</strong> Notes in Progress</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split("#")[0].split("?")[0];
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
