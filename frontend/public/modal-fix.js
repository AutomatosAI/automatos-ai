// Force close any stuck modals on page load
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(() => {
    // Find any open dialogs and close them
    const openDialogs = document.querySelectorAll('[role="dialog"][data-state="open"]');
    openDialogs.forEach(dialog => {
      const closeBtn = dialog.querySelector('[data-radix-collection-item]') || 
                      dialog.querySelector('button[aria-label="Close"]') ||
                      dialog.querySelector('.absolute.right-4.top-4');
      if (closeBtn) closeBtn.click();
    });
    
    // Remove any blocking overlays
    const overlays = document.querySelectorAll('.fixed.inset-0.z-\[100\]');
    overlays.forEach(overlay => {
      if (overlay.getAttribute('aria-hidden') === 'true') {
        overlay.style.pointerEvents = 'none';
      }
    });
  }, 1000);
});

// Add ESC key handler to force close modals
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    const openDialogs = document.querySelectorAll('[role="dialog"][data-state="open"]');
    openDialogs.forEach(dialog => {
      dialog.style.display = 'none';
    });
    const overlays = document.querySelectorAll('.fixed.inset-0');
    overlays.forEach(overlay => overlay.remove());
  }
});
