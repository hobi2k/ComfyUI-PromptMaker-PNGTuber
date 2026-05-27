import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function chainCallback(object, property, callback) {
    const original = object?.[property];
    object[property] = function (...args) {
        const result = original?.apply(this, args);
        return callback.apply(this, args) ?? result;
    };
}

async function getAuthHeader() {
    try {
        const authStore = await api.getAuthStore?.();
        return authStore ? await authStore.getAuthHeader() : null;
    } catch {
        return null;
    }
}

async function uploadVideo(file, progressCallback) {
    const body = new FormData();
    body.append("image", new File([file], file.name, { type: file.type, lastModified: file.lastModified }));
    const response = await new Promise((resolve) => {
        const request = new XMLHttpRequest();
        request.upload.onprogress = (event) => progressCallback?.(event.loaded / event.total);
        request.onload = () => resolve(request);
        request.open("POST", api.apiURL("/upload/image"), true);
        getAuthHeader().then((headers) => {
            for (const key in headers ?? {}) {
                request.setRequestHeader(key, headers[key]);
            }
            request.send(body);
        });
    });
    if (response.status !== 200) {
        throw new Error(`${response.status} - ${response.statusText}`);
    }
    return JSON.parse(response.responseText).name;
}

function setWidgetValue(widget, filename) {
    if (widget.options?.values && !widget.options.values.includes(filename)) {
        widget.options.values.push(filename);
    }
    widget.value = filename;
    widget.callback?.(filename);
}

function addVideoUpload(nodeType, widgetName) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const widget = this.widgets?.find((item) => item.name === widgetName);
        if (!widget) {
            return;
        }

        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/webm,video/mp4,video/x-matroska,video/quicktime,image/gif";
        input.style.display = "none";
        document.body.append(input);

        chainCallback(this, "onRemoved", () => input.remove());

        const runUpload = async (file) => {
            if (!file) {
                return false;
            }
            try {
                const filename = await uploadVideo(file, (progress) => {
                    this.progress = progress;
                });
                setWidgetValue(widget, filename);
                return true;
            } catch (error) {
                alert(error?.message ?? error);
                return false;
            } finally {
                this.progress = undefined;
            }
        };

        input.onchange = async () => {
            await runUpload(input.files?.[0]);
            input.value = "";
        };

        this.onDragOver = (event) => !!event?.dataTransfer?.types?.includes?.("Files");
        this.onDragDrop = async (event) => {
            if (!event?.dataTransfer?.types?.includes?.("Files")) {
                return false;
            }
            return await runUpload(event.dataTransfer.files?.[0]);
        };

        const uploadWidget = this.addWidget("button", "upload video", null, () => {
            app.canvas.node_widget = null;
            input.click();
        });
        uploadWidget.options.serialize = false;
    });
}

app.registerExtension({
    name: "PromptMaker.PNGTuber.VideoUpload",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "PNGTuberVideoUploadToMouthBundle") {
            addVideoUpload(nodeType, "video");
        }
        if (nodeData?.name === "PNGTuberVideoMouthBuilder") {
            addVideoUpload(nodeType, "video_path");
        }
    },
});
