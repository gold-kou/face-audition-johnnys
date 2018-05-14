/**
 * デフォルトでボタンをdisabledにする関数
 **/
window.onload = function () {
    document.getElementById("btnUpload").disabled = true;
}

/**
 * 画像アップロード済みであればボタンのdisabledを解除する関数
 **/
function selectFile() {
    if (document.getElementById("uploadFile").value === "") {
        document.getElementById("btnUpload").disabled = true;
    } else {
        document.getElementById("btnUpload").disabled = false;
    }
}

/**
 * アップロードした画像のプレビューを実行する関数
 **/
$(function () {
    //画像ファイルプレビュー表示のイベント追加 fileを選択時に発火するイベントを登録
    $('form').on('change', 'input[type="file"]', function (e) {
        var file = e.target.files[0],
            reader = new FileReader(),
            $preview = $(".imagePreview");
        t = this;

        // 画像ファイル以外の場合は何もしない
        if (file.type.indexOf("image") < 0) {
            return false;
        }

        // ファイル読み込みが完了した際のイベント登録
        reader.onload = (function (file) {
            return function (e) {
                //既存のプレビューを削除
                $preview.empty();
                // .prevewの領域の中にロードした画像を表示するimageタグを追加
                $preview.append($('<img>').attr({
                    src: e.target.result,
                    class: "imagePreview",
                    title: file.name
                }));
            };
        })(file);

        reader.readAsDataURL(file);
    });
});
