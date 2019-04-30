<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">

<!-- jQuery library -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>

<!-- Latest compiled JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
</head>
<body>
<div class="container">
    <div class="row" style="margin-top:40px;">
        <div class="col-sm-5"></div>
          <div class="col-sm-4"><b><u>Query Image:</u></b></div>
          <div class="col-sm-3"><a href="back.php"><b><u>Return to Query Page</u></b></a></div>
    </div>
    <br>
    <div class="row" >
        <div class="col-sm-5"></div>
        <div class="col-sm-4">
        <?php
            $query_image=glob("uploads/*.*");
            for($i=0;$i<count($query_image);$i++)
            {
                $image = $query_image[$i];
                $supported_file = array(
                        'gif',
                        'jpg',
                        'jpeg',
                        'png'
                );

                $ext = strtolower(pathinfo($image, PATHINFO_EXTENSION));
                if (in_array($ext, $supported_file)) { // show only image name if you want to show full path then use this code // echo $image."<br />";
                    ?><img src="<?php echo $image ?>" alt="Random image"  height="140px" width="140px" halign="middle"/><?php
                    } else {
                        continue;
                    }
          }?>
        </div>
    </div>
    <br><br/><br/><br/>
    <div class="row">
          <div class="col-sm-12"><b><u>Result Images:</u></b></div>
    </div>
        <br/>
        <div class="row">
          <?php
     $files = glob("resultimages/*.*");
     for ($i=0; $i<count($files); $i++)
      {
        $image = $files[$i];
        $supported_file = array(
                'gif',
                'jpg',
                'jpeg',
                'png'
         );

         $ext = strtolower(pathinfo($image, PATHINFO_EXTENSION));
         if (in_array($ext, $supported_file)) { // show only image name if you want to show full path then use this code // echo $image."<br />";
             echo '<img src="'.$image .'" alt="Random image"  height="150px" width="150px"/>'."  ";
            } else {
                continue;
            }
          }
       ?>
    </div>
</div>    
</body>
</html>
