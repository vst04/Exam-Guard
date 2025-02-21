const menuBtn = document.getElementById("menu-btn");
const navLinks = document.getElementById("nav-links");

if (menuBtn && navLinks) {
  const menuBtnIcon = menuBtn.querySelector("i");

  menuBtn.addEventListener("click", () => {
    navLinks.classList.toggle("open");

    const isOpen = navLinks.classList.contains("open");
    if (menuBtnIcon) {
      menuBtnIcon.setAttribute("class", isOpen ? "ri-close-line" : "ri-menu-line");
    }
  });

  navLinks.addEventListener("click", () => {
    navLinks.classList.remove("open");
    if (menuBtnIcon) {
      menuBtnIcon.setAttribute("class", "ri-menu-line");
    }
  });
}

// ScrollReveal options
const scrollRevealOption = {
  distance: "50px",
  origin: "bottom",
  duration: 1000,
};

if (typeof ScrollReveal !== "undefined") {
  ScrollReveal().reveal(".header__image img", {
    ...scrollRevealOption,
    origin: "right",
  });
  ScrollReveal().reveal(".header__content h2", {
    ...scrollRevealOption,
    delay: 500,
  });
  ScrollReveal().reveal(".header__content h1", {
    ...scrollRevealOption,
    delay: 1000,
  });
  ScrollReveal().reveal(".header__content p", {
    ...scrollRevealOption,
    delay: 1500,
  });
  ScrollReveal().reveal(".header__content .header__btn", {
    ...scrollRevealOption,
    delay: 2000,
  });
  ScrollReveal().reveal(".header__content .socials", {
    ...scrollRevealOption,
    delay: 2500,
  });
  ScrollReveal().reveal(".header__bar", {
    ...scrollRevealOption,
    delay: 3000,
  });
}

// Password visibility toggle
const passwordToggle = (inputId, eyeId) => {
  const input = document.getElementById(inputId);
  const eyeIcon = document.getElementById(eyeId);

  if (!input || !eyeIcon) return;

  eyeIcon.addEventListener("click", () => {
    input.type = input.type === "password" ? "text" : "password";
    eyeIcon.classList.toggle("ri-eye-fill");
    eyeIcon.classList.toggle("ri-eye-off-fill");
  });
};

// Video feed error handling function
function handleVideoError(img) {
  console.error("Video feed error");
  // Retry connection after 5 seconds
  setTimeout(() => {
      img.src = "http://localhost:5000/video_feed?" + new Date().getTime();
  }, 5000);
}
// Add this to your existing script
document.getElementById('video-feed').onerror = function() {
  handleVideoError(this);
};

document.addEventListener("DOMContentLoaded", function () {
  const loginContainer = document.getElementById("loginAccessRegister");
  const registerBtn = document.getElementById("loginButtonRegister");
  const loginBtn = document.getElementById("loginButtonAccess");
  const getStartedBtn = document.getElementById("getStartedButton");
  
  // Initialize video feed
  const videoFeed = document.querySelector('.camera-feed img');
  if (videoFeed) {
      videoFeed.src = "http://localhost:5000/video_feed?" + new Date().getTime();
  }

  // Show signup form when "Create Account" is clicked
  if (registerBtn && loginContainer) {
    registerBtn.addEventListener("click", () => {
      loginContainer.classList.add("active");
    });
  }

  // Show login form when "Log In" is clicked
  if (loginBtn && loginContainer) {
    loginBtn.addEventListener("click", () => {
      loginContainer.classList.remove("active");
    });
  }

  // Show signup form when "Get Started" is clicked
  if (getStartedBtn && loginContainer) {
    getStartedBtn.addEventListener("click", () => {
      loginContainer.classList.add("active");
    });
  }
});

// Additional ScrollReveal Animations
if (typeof ScrollReveal !== "undefined") {
  ScrollReveal().reveal(".login__title", {
    ...scrollRevealOption,
    delay: 100,
  });

  ScrollReveal().reveal(".login__box", {
    ...scrollRevealOption,
    delay: 200,
    interval: 100,
  });

  ScrollReveal().reveal(".login__button", {
    ...scrollRevealOption,
    delay: 400,
  });

  ScrollReveal().reveal(".login__social", {
    ...scrollRevealOption,
    delay: 600,
  });

  ScrollReveal().reveal(".login__switch", {
    ...scrollRevealOption,
    delay: 800,
  });
}