// Deep Learning from Scratch in Java - Book JavaScript

(function() {
    'use strict';

    // DOM Elements
    const sidebar = document.querySelector('.sidebar');
    const menuToggle = document.querySelector('.menu-toggle');
    const chapterContainer = document.querySelector('.chapter-container');

    // Mobile menu toggle
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 1024 &&
                !sidebar.contains(e.target) &&
                !menuToggle.contains(e.target) &&
                sidebar.classList.contains('open')) {
                sidebar.classList.remove('open');
            }
        });
    }

    document.addEventListener('click', function(event) {
        const btn = event.target.closest('.copy-btn');
        if (!btn) return;

        const pre = btn.closest('pre') || btn.closest('.code-block')?.querySelector('pre')
            || btn.parentElement.nextElementSibling;
        const code = pre?.querySelector('code');
        if (!code) return;

        const text = code.innerText;
        const copied = function() {
            const originalText = btn.textContent;
            btn.textContent = 'Copied!';
            btn.style.background = '#27ae60';
            btn.style.color = '#fff';
            setTimeout(function() {
                btn.textContent = originalText;
                btn.style.background = '';
                btn.style.color = '';
            }, 2000);
        };
        const fallback = function() {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            copied();
        };
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(text).then(copied).catch(fallback);
        } else {
            fallback();
        }
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const target = document.querySelector(targetId);
            if (target) {
                e.preventDefault();
                const headerHeight = document.querySelector('.top-header')?.offsetHeight || 0;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - headerHeight - 20;
                window.scrollTo({ top: targetPosition, behavior: 'smooth' });
                
                // Close mobile menu if open
                if (sidebar.classList.contains('open')) {
                    sidebar.classList.remove('open');
                }
            }
        });
    });

    // Highlight active chapter in sidebar based on scroll position
    function highlightActiveChapter() {
        const chapters = document.querySelectorAll('.chapter-container h2[id], .chapter-container h3[id]');
        const navLinks = document.querySelectorAll('.nav-chapter');
        const headerHeight = document.querySelector('.top-header')?.offsetHeight || 60;
        
        let currentId = '';
        
        chapters.forEach(function(chapter) {
            const rect = chapter.getBoundingClientRect();
            if (rect.top <= headerHeight + 50) {
                currentId = chapter.id;
            }
        });

        if (!currentId || !Array.from(navLinks).some(function(link) {
            return link.getAttribute('href') === '#' + currentId;
        })) return;

        navLinks.forEach(function(link) {
            link.classList.toggle('active', link.getAttribute('href') === '#' + currentId);
        });
    }

    // Debounced scroll handler
    let scrollTimeout;
    window.addEventListener('scroll', function() {
        if (scrollTimeout) clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(highlightActiveChapter, 50);
    });

    // Initial highlight
    highlightActiveChapter();

    // Add copy buttons to all code blocks
    function addCopyButtons() {
        document.querySelectorAll('pre').forEach(function(pre) {
            if (pre.querySelector('.copy-btn') || pre.previousElementSibling?.classList.contains('code-header')) return;
            
            const header = document.createElement('div');
            header.className = 'code-header';
            header.innerHTML = '<span class="code-lang">Java</span><button class="copy-btn" title="Copy to clipboard">Copy</button>';
            pre.insertBefore(header, pre.firstChild);
        });
    }

    // Run after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addCopyButtons);
    } else {
        addCopyButtons();
    }

    // Keyboard navigation
    document.addEventListener('keydown', function(e) {
        // Left/Right arrow keys for chapter navigation
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        
        const prevLink = document.querySelector('.nav-link.prev');
        const nextLink = document.querySelector('.nav-link.next');
        
        if (e.key === 'ArrowLeft' && prevLink) {
            prevLink.click();
        } else if (e.key === 'ArrowRight' && nextLink) {
            nextLink.click();
        }
        
        // Escape to close mobile menu
        if (e.key === 'Escape' && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
        }
    });

    // Reading progress indicator
    function createProgressBar() {
        const progressBar = document.createElement('div');
        progressBar.className = 'reading-progress';
        progressBar.style.cssText = `
            position: fixed;
            top: ${document.querySelector('.top-header')?.offsetHeight || 60}px;
            left: 0;
            height: 3px;
            background: linear-gradient(90deg, #3498db, #2980b9);
            z-index: 999;
            transform-origin: left;
            transform: scaleX(0);
            pointer-events: none;
        `;
        document.body.appendChild(progressBar);
        
        window.addEventListener('scroll', function() {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const progress = scrollTop / docHeight;
            progressBar.style.transform = 'scaleX(' + progress + ')';
        });
    }

    // Initialize progress bar
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', createProgressBar);
    } else {
        createProgressBar();
    }

    // Add IDs to headings for navigation
    function addHeadingIds() {
        document.querySelectorAll('.chapter-container h2, .chapter-container h3, .chapter-container h4').forEach(function(heading, index) {
            if (!heading.id) {
                const text = heading.textContent.trim().toLowerCase()
                    .replace(/[^\w\s-]/g, '')
                    .replace(/\s+/g, '-');
                heading.id = text || 'heading-' + index;
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addHeadingIds);
    } else {
        addHeadingIds();
    }

})();