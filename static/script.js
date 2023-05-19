const selectImage = document.querySelector('.select-image');
const inputFile = document.querySelector('#file');
const imgArea = document.querySelector('.img-area');

selectImage.addEventListener('click', function () {
	inputFile.click();
})

inputFile.addEventListener('change', function () {
	const image = this.files[0]
	if(image.size < 2000000) {
		const reader = new FileReader();
		reader.onload = ()=> {
			const allImg = imgArea.querySelectorAll('img');
			allImg.forEach(item=> item.remove());
			const imgUrl = reader.result;
			const img = document.createElement('img');
			img.src = imgUrl;
			imgArea.appendChild(img);
			imgArea.classList.add('active');
			imgArea.dataset.img = image.name;
		}
		reader.readAsDataURL(image);
	} else {
		alert("Image size more than 2MB");
	}
})
// const btn = document
//     .querySelector('.btn');
// const btnText = document
//     .querySelector('btn.text');p

// const progress = document
//     .querySelector('.progress');

// document
//     .querySelector('upload-btn')
//     .addEventListener('click', ()=> {
     
//         btn.classList.remove('toggle--upload');
//         btnText.classList.remove('toggle--upload-text');
//         progress.classList.remove('toggle--progress');

//         setTimeout(()=>{
//             btn.classList.add('toggle-upload');
//             btnText.classList.add(toggle-progress);
//             progress.classList.add('toggle--progress');
//         },50);

//         setTimeout(()=>{
//             btnText.textContent='uploding...'
//         },300);

//     });

//     btn.addEventListener('webkitAnimationEnd',()=>{
//         setTimeout(() =>{
//             btnText.textContent ='upload';


//         },2000);

//     });