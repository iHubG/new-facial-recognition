/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/views/*.html"],
  theme: {
    extend: {
      backgroundImage: {
        'face-recognition': "url('/static/img/face-recognition.jpg')",
      },
    },
  },
  plugins: [
    require('daisyui'),
  ],
}

