function offset(element) {
  const box = element.getBoundingClientRect();
  return {
    top: box.top + (window.scrollY || document.documentElement.scrollTop) - (document.documentElement.clientTop || 0),
    left: box.left + (window.scrollX || document.documentElement.scrollLeft) - (document.documentElement.clientLeft || 0)
  }
}

export default function buttonAnimation() {
  /* ==  float_box_container button  == */
  const float_box_container = document.querySelectorAll(".float_box_container");

  if (float_box_container.length) {
    float_box_container.forEach(button => {
      button.addEventListener('mousemove', (e) => {
        const float_box = button.querySelector(".float_box");

        let parentOffset = offset(button);
        let relX = e.pageX - parentOffset.left;
        let relY = e.pageY - parentOffset.top;

        float_box.style.left = relX + 'px';
        float_box.style.top = relY + 'px';
        float_box.classList.add('show');
      });

      button.addEventListener('mouseleave', (e) => {
        const float_box = button.querySelector(".float_box");
        float_box.classList.remove('show');
      });
    })
  }

  /* ==  Button Animation  == */
  const buttonAnimation = document.querySelectorAll(".button_su_inner");

  if (buttonAnimation.length) {
    buttonAnimation.forEach(button => {
      button.addEventListener('mousemove', (e) => {
        const button_circle = button.parentElement.querySelector(".su_button_circle");

        if (button_circle) {
          let parentOffset = offset(button);
          let relX = e.pageX - parentOffset.left;
          let relY = e.pageY - parentOffset.top

          button_circle.style.left = relX + 'px';
          button_circle.style.top = relY + 'px';
          button_circle.classList.remove("desplode-circle");
          button_circle.classList.add("explode-circle");
        }
      });

      button.addEventListener('mouseleave', (e) => {
        const button_circle = button.parentElement.querySelector(".su_button_circle");

        if (button_circle) {
          let parentOffset = offset(button);
          let relX = e.pageX - parentOffset.left;
          let relY = e.pageY - parentOffset.top

          button_circle.style.left = relX + 'px';
          button_circle.style.top = relY + 'px';
          button_circle.classList.add("desplode-circle");
          button_circle.classList.remove("explode-circle");
        }
      });
    })
  }
}