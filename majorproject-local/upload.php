<?php
    
    session_start();
    if(isset($_POST['upload_submit']))
    {
        if ($_FILES["file"]["error"][0] > 0)
        {
            echo "Error: " . $_FILES["file"]["error"][0] . "<br>";
        }
        else
        {

            $total = count($_FILES["file"]["name"]);
            // Loop through each file
            for( $i=0 ; $i < $total ; $i++ ) 
            {
                //Get the temp file path
                $tmpFilePath = $_FILES["file"]["tmp_name"][$i];

                //Make sure we have a file path
                if ($tmpFilePath != "")
                {
                //Setup our new file path
                    $newFilePath = "images\\" . $_FILES["file"]["name"][$i];
                    move_uploaded_file($tmpFilePath, $newFilePath);
                }
            }
        }
    }
    else if(isset($_POST['run_submit']))
    {
        shell_exec("python ins.py");
        //header("Location: upload.php");
    }
    else
    {
        $_SESSION['login_successful']=$_SESSION['login_successful']?$_SESSION['login_successful']:false;
        if(!$_SESSION['login_successful'])
            header("Location: login.html");
    }

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link href="//netdna.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
    <script src="//netdna.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js"></script>
    <script src="//code.jquery.com/jquery-1.11.1.min.js"></script>
    <link rel="stylesheet" href="upload.css">
    <script src="upload.js"></script></script>
</head>
<body>

<div class="container">
        <div class="panel panel-default">
            <div class="panel-heading"><strong>Upload Files</strong> <small>(Image files upload)</small><span class="badge alert-success pull-right"><a href="logout.php">logout</a></span></div>
        <div class="panel-body">

            <!-- Standar Form -->
            <h4>Select files from your computer</h4>
            <form action="upload.php" method="post" enctype="multipart/form-data" id="js-upload-form">
            <div class="form-inline">
                <div class="form-group">
                <input type="file" name="file[]" id="js-upload-files" multiple>
                </div>
                <button type="submit" class="btn btn-sm btn-primary" id="js-upload-submit" name="upload_submit">Upload files</button>
                <button type="submit" class="btn btn-sm btn-primary" id="js-python-submit" name="run_submit">RUN</button>
            </div>
            </form>

            <!-- Drop Zone -->
            <h4>Or drag and drop files below</h4>
            <div class="upload-drop-zone" id="drop-zone">
            Just drag and drop files here
                <div>
                <i class="fas fa-spinner"></i>
                </div>
            </div>

            <!-- Progress Bar -->
            <div class="progress">
            <div class="progress-bar" role="progressbar" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100" style="width: 60%;">
                <span class="sr-only">60% Complete</span>
            </div>
            </div>

            <!-- Upload Finished -->
            <div class="js-upload-finished">
            <h3>Processed files</h3>
            <div class="list-group">

            <?php
                if(isset($_POST['upload_submit']))
                {
                $total = count($_FILES["file"]["name"]);
                for( $i=0 ; $i < $total ; $i++ )
                {
                    $tmpFilePath = $_FILES["file"]["name"][$i];
                ?>
                    <a href="#" class="list-group-item list-group-item-success"><span class="badge alert-success pull-right">Success</span>
                <?php
                
                    echo $tmpFilePath;
                
                }}
                ?>
                </a>
            </div>
            </div>
        </div>
        </div>
    </div> <!-- /container -->
</body>
</html>