// Navigation sidebar functionality
document.querySelectorAll('.sidebar a').forEach(link => {
  link.addEventListener('click', e => {
      e.preventDefault();
      const sectionId = link.getAttribute('data-section');
      
      // Hide all sections
      document.querySelectorAll('.section').forEach(section => {
          section.classList.remove('active');
      });
      
      // Show selected section
      document.getElementById(sectionId).classList.add('active');
      
      // Update active link
      document.querySelectorAll('.sidebar a').forEach(a => {
          a.classList.remove('active');
      });
      link.classList.add('active');
  });
});